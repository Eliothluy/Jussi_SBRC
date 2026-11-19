# Relatório de Validação — Etapas 1 e 2 (`IPC Handshake v2`, `BUG-03`, `BUG-04`, `BUG-05`)

> **Projeto:** `Jussi_SBRC` / `MODA` · **Branch ativa:** `exp/sqr-v2-infra` · **Data:** `2026-11-17`  
> **Escopo:** Conclusão das **Etapas 1 e 2** de [`PLANO_CORRECAO_INFRAESTRUTURA_V2.md`](../PLANO_CORRECAO_INFRAESTRUTURA_V2.md) (`v1.1`), com **PARADA OBRIGATÓRIA** antes da **Etapa 3A** (`BUG-01` / `BUG-06`). Nenhuma seed confirmatória (`MAIN 3001..3010`, `G4B 1201..1209`, `OOS >= 3101`) e nenhum controlador novo (`SQR-CER`, DRL ou bandit) foram implementados ou executados.

---

## 1. Isolamento de Versão e Preservação da `v1` (Etapa 1)

1. **Auditoria pré-commit e preservação de estado preexistente:**
   - Registrada em [`ETAPA1_INVENTARIO_PRE_COMMIT.md`](ETAPA1_INVENTARIO_PRE_COMMIT.md) (`1` arquivo modificado + `20` não rastreados inspecionados com tamanho e SHA-256).
   - Commit documental local na `main`: `168c503d46cbb6558a83d78bbf242f4b91f80b93` (incluindo os documentos da auditoria, relatórios de decisão e [`PLANO_CORRECAO_INFRAESTRUTURA_V2.md`](../PLANO_CORRECAO_INFRAESTRUTURA_V2.md) `v1.1`).
   - Tag local anotada criada: `v1.0-b0-frozen-audit-2026-11-17` (`ce61b5ebb8e60aecfacbeca4db97a9ac42a51c2c`).
   - Branch local criada e ativa para o desenvolvimento isolado `v2`: `exp/sqr-v2-infra`.
   - **Remoto `origin/main` completamente intocado:** `060d0581f950d041fb5c4f6482259528089cf019` (zero `git push` ou alterações em branches remotas).
   - Arquivo preexistente [`Auditoria.txt`](../../superpowers/specs/Auditoria.txt) preservado byte a byte fora de commit (`SHA-256`: `cfe517c6efc96691d2630d9b72526eb681b460b95e8772b90c9acd5ebe177924`).
2. **Integridade do congelamento `B0` verificada após a edição final:**
   - [`verify_b0_frozen_manifest`](../../../ns-o-ran-gym/src/nsoran/sqr_v2/manifest_v2.py#L67-L107) contra [`B0_FROZEN_MANIFEST.json`](../jc_decision/B0_FROZEN_MANIFEST.json):
     - `git_provenance.code_sha256`: **`17/17` correspondências exatas** (`all_ok: true`);
     - `verified_dev_checkpoints`: **`5/5` correspondências exatas**;
     - Binário `v1` congelado [`ns3.46-sqr-sim-default`](../../../ns-3-dev/build/scratch/sqr/ns3.46-sqr-sim-default): `6403e2ebdc7a7318adf18f37acb674134a9fc412421c3cbb0dd1d5528c9eb0b1`;
     - Binário `v2` isolado [`ns3.46-sqr-v2-sim-default`](../../../ns-3-dev/build/scratch/sqr_v2/ns3.46-sqr-v2-sim-default): `8a19968192df6f67e74fee30a5e0df76261a741bbd7989afc68470f5a58e97f5`;
     - Pasta histórica [`results/sqr_campaign/tables/`](../../../ns-o-ran-gym/results/sqr_campaign/tables/) preservada sem alterações (mtimes originais de `2026-10-13 20:48`).
3. **Governança de seeds (`v2`):** [`validate_dev_seed`](../../../ns-o-ran-gym/src/nsoran/sqr_v2/manifest_v2.py#L53-L64) bloqueia programaticamente `3001..3010` (`MAIN`), `1201..1209` (`G4B`) e `>= 3101` (`OOS`) com `ValueError`.
4. **Regressão completa da suíte legada `v1` após a última edição:**
   - Comando: `bash MODA/ns-o-ran-gym/tests/run_sqr_tests.sh -x -q`
   - Resultado: **`360 passed, 2 warnings in 56.93s`** (`exit 0`).

---

## 2. Arquitetura do `IPC Handshake v2` (`BUG-03` + `BUG-05`)

Implementada de forma isolada em [`sqr-v2-sim.cc`](../../../ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc) (C++ ns-3/5G-LENA) e em [`nsoran.sqr_v2`](../../../ns-o-ran-gym/src/nsoran/sqr_v2/__init__.py) ([`ipc_v2.py`](../../../ns-o-ran-gym/src/nsoran/sqr_v2/ipc_v2.py), [`env_v2.py`](../../../ns-o-ran-gym/src/nsoran/sqr_v2/env_v2.py), [`controller_v2.py`](../../../ns-o-ran-gym/src/nsoran/sqr_v2/controller_v2.py), [`manifest_v2.py`](../../../ns-o-ran-gym/src/nsoran/sqr_v2/manifest_v2.py)) + driver [`sqr_v2_run.py`](../../../ns-o-ran-gym/examples/sqr_v2_run.py):

1. **Identificação inequívoca por `run_id` e `window_seq`:**
   - O ns-3 escreve atomicamente (`.tmp -> rename`) [`sqr_ipc_indication_v2.json`](../../../ns-o-ran-gym/src/nsoran/sqr_v2/ipc_v2.py#L11-L86) com `protocol_version=2`, `run_id`, `window_seq`, `sim_time_ms`, `period_ms`, `total_rbg`, `total_prb`, `active_ues` e recibo físico da janela recém-encerrada (`governed_prev_window_seq`, `prev_applied_action_seq`, `prev_ipc_status`, `prev_prb_executed_rbg`, `prev_decision_wall_ms`, `recent_windows_tail` com até 8 janelas).
   - O Python escreve atomicamente (`write_action_v2_atomic`, `.tmp.<pid> -> os.replace`) [`sqr_ipc_action_v2.csv`](../../../ns-o-ran-gym/src/nsoran/sqr_v2/ipc_v2.py#L115-L162) contendo `run_id,window_seq,sim_time_ms,slice_id,prb_proposed,prb_requested,prb_allocated,dedicated_pct,algorithm,agent_decision_wall_ms` terminado obrigatoriamente pelo sentinela `# END_V2`.
2. **Ordem de flush e remoção de corrida/consumo obsoleto:**
   - Em [`KpmAndControlCallback`](../../../ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc#L1880-L2260), `sqr_prb_windows.csv` (com `fflush`), `sqr-kpms.txt.tmp -> rename` e `FlushPacketDelays()` são concluídos **antes** de `WriteIndicationEnvelopeV2(...)`, `DrainStaleControlBeforePost(currentWindowSeq)` e `sem_post(g_semMetricsReady)`.
   - Antes do post e imediatamente após aplicar uma ação válida, [`DrainStaleControlBeforePost`](../../../ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc#L1738-L1768) e [`DrainImmediateDuplicatesAfterApply`](../../../ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc#L1770-L1804) consomem tokens/arquivos remanescentes no semáforo `g_semControl` para que nenhum token atrasado seja confundido com a janela seguinte.
   - O consumo pelo C++ usa `std::filesystem::rename` atômico para `.processing` antes do parsing.
3. **Política explícita de timeout (`TIMEOUT_HOLD`) e aprendizado causal estritamente sobre `prb_executed_mac`:**
   - Ao estourar `ipcTimeoutMs` (padrão `50 ms` via `AddRealtimeDeadlineMs`), o C++ **mantém** a configuração ativa anterior no MAC (`heldQuota = ResolveActiveMacQuotaRbg()`), registra `ipc_status = "TIMEOUT_HOLD"`, `applied_action_seq = g_lastAppliedActionSeq`, `executedRbg = heldQuota` e `waitedMs`.
   - Em [`reconcile_prev_receipt`](../../../ns-o-ran-gym/src/nsoran/sqr_v2/ipc_v2.py#L165-L226) e [`run_v2_episode`](../../../ns-o-ran-gym/examples/sqr_v2_run.py#L216-L263), cada estado `s_t` (`window_seq = w`) é conciliado com o recibo de `governed_prev_window_seq == w` na fronteira `w + 1`:
     - `controllers[sid].observe(states[sid], int(prb_executed_mac[sid]), float(y_t))` usa **exclusivamente `prb_executed_mac`** (jamais a proposta rejeitada);
     - `apply_virtual_updates(...)` só executa quando `ipc_status == "APPLIED"` e `window_gap == 0` (em `TIMEOUT_HOLD`, `virtual_updates == 0`);
     - quando `window_gap != 0`, o aprendizado é explicitamente bloqueado (`can_learn == False`).
4. **Sincronização inicial de capacidade `C = 53` antes do Step 1 (`BUG-05`):**
   - `sqr-v2-sim` exporta `total_rbg` e `total_prb` (`53` RBGs para `10 MHz`, `mu=0`, `300 kHz` RBG) já na primeira indicação (`env.reset()`), permitindo a sincronização `n_prbs = 53` nos controladores antes da primeira decisão (`_sync_controllers_capacity`).
   - Em `test_bug05_step1_capacity_sync_equal_controller_real_ns3`, o Step 1 executa `[18, 18, 17]` (`soma = 53`) e o Passo 2 executa `[17, 18, 18]` (`soma = 53`), sem capacidade stale `C = 52`.

---

## 3. Testes de Injeção de Falhas (5 Classes) e Comportamento sob Timeouts (`50 ms`)

Executados integralmente contra o simulador ns-3 real (`ns3.46-sqr-v2-sim-default`) na suíte [`test_sqr_v2_ipc_and_metrics.py`](../../../ns-o-ran-gym/tests/test_sqr_v2_ipc_and_metrics.py) com `--ipcTimeoutMs=50`:

| Classe de falha | Cenário injetado nos testes | Mecanismo no C++/Python | Verificação observada (`PASSED`) |
|---|---|---|---|
| **1. Obsoleta** (`window_seq < w`) | `seq == 6`: envio prévio de payload `window_seq = 4` (`t=400 ms`) | `ParseAndValidateActionCsvV2` retorna `OBSOLETE`, descarta e aguarda resposta válida de `w=6` | `obsolete_discarded_count >= 1`, `applied_count == windows_total`, `window_seq == applied_action_seq` |
| **2. Duplicada** (`window_seq == lastApplied`) | **2a (`seq == 8`):** reenvio com `window_seq = 7`; **2b (`seq == 10`):** proxy `_DoubleSendSameSeqProxy` reenvia o próprio `window_seq = 10` após a aplicação | Check `parsed.windowSeq == g_lastAppliedActionSeq` antes de `OBSOLETE`, tanto no loop de espera quanto em `DrainImmediateDuplicatesAfterApply` / `DrainStaleControlBeforePost` | `duplicate_discarded_count >= 2`, zero reaplicação espúria, `timeout_hold_count == 0` |
| **3. Atrasada** (`> 50 ms`) | `test_ipc_v2_delays_single_alternating_burst_and_mac_learning_only`: atraso de `75 ms` (`time.sleep(0.075)`) com proposta forçada `41 PRBs` no slice 0 em janela **isolada** (`25`), **alternada** (`30, 32`) e **rajada de 3 passos** (`40, 41, 42`) | C++ expira em `50 ms`, grava `TIMEOUT_HOLD`, drena os tokens atrasados na fronteira seguinte; Python aprende apenas sobre `prb_executed_mac != 41` com `virtual_updates == 0` | `timeout_hold_count >= 6`, `learned_on_hold >= 1`, recuperação imediata com **zero deslocamento** em `26, 31, 33, 43` (`ipc_status == "APPLIED"` e `applied_action_seq == clean_next`), e nenhuma entrada `41 / 53` em `ctls[0].regressor.support_points[:, -1]` |
| **4. Incompleta / Malformada** | `seq == 4`: (4a) truncada sem `# END_V2`, (4b) apenas 2 slices com `# END_V2`, (4c) valor `nan` em `dedicated_pct` | `ParseAndValidateActionCsvV2` exige `# END_V2`, cabeçalho, `3` slices únicos `[0..2]` e floats finitos estritos (`ParseStrictFiniteDouble`), retornando `MALFORMED` | `malformed_discarded_count >= 3`, aplicação preservada em `w=4` |
| **5. Outra execução** (`run_id` distinto) | `seq == 2`: payload válido com `run_id="foreign_run_999"` | `parsed.runId != g_runId` retorna `WRONG_RUN` | `wrong_run_discarded_count >= 1`, sem contaminação entre execuções |

---

## 4. Paridade `A/A` em Malha Fechada e Separação entre Validação Causal e Desempenho Temporal (`50 ms`)

1. **Separação explícita nos manifestos (`SQR_V2_MANIFEST.json`) e CLI:**
   - [`build_v2_manifest`](../../../ns-o-ran-gym/src/nsoran/sqr_v2/manifest_v2.py#L128-L200) separa:
     - **`strict_rt`** (`--ipc-mode strict_rt --ipc-timeout-ms 50`): `rt_50ms_deadline_enforced = True`; utilizado para **prova de conformidade temporal `<= 50 ms`** e testes com deadline real.
     - **`causal_sim`** (`--ipc-mode causal_sim`): `rt_50ms_deadline_enforced = False`; reservado para validação causal/semântica de longo prazo, explicitamente **proibido como prova de desempenho RT em `50 ms`**.
   - Verificado em `test_causal_vs_rt_separation_in_manifest` (`ipc_mode="causal_sim"` com `ipc_timeout_ms=2000` gera `rt_50ms_deadline_enforced is False`, enquanto `ipc_mode="strict_rt"` com `ipc_timeout_ms=50` gera `rt_50ms_deadline_enforced is True`; em [`manifest_v2.py`](../../../ns-o-ran-gym/src/nsoran/sqr_v2/manifest_v2.py#L148), a condição exigida é `ipc_mode == "strict_rt" and int(ipc_timeout_ms) <= 50`).
2. **Paridade `A/A` em malha fechada sob deadline real de `50 ms` (`test_closed_loop_aa_parity_strict_rt_real_ns3`):**
   - Duas execuções independentes `A1` e `A2` (`--controller sqr --gamma 1 --seed 101 --simTime 10.0 --cellRadius 300 --sliceLambdas 12,12,12 --warmup_s 2.0 --ipc-mode strict_rt --ipc-timeout-ms 50`):
   - **Paridade bit-a-bit confirmada:** `step_metrics.csv` idêntico em todas as colunas determinísticas (`pd.testing.assert_frame_equal(..., check_exact=True)`) e SHA-256 idêntico dos checkpoints (`ck_a["meta"]["payload_sha256"] == ck_b["meta"]["payload_sha256"]`), com `timeout_hold_count == 0` e `applied_count == windows_total`.
   - **Telemetria observada na execução com `-s` (`task-3262`, mesmo binário C++ e controlador `v2`; revalidado em `task-3343` após a correção apenas no teste de injeção de falhas):**
     - `aa_parity_A1`: `steps=98`, `applied=98`, `holds=0`, `agent_decision_wall_ms`: `p50 = 0.897 ms`, `p95 = 1.222 ms`, `max = 1.439 ms`, `ns3_exit = 0`.
     - `aa_parity_A2`: `steps=98`, `applied=98`, `holds=0`, `agent_decision_wall_ms`: `p50 = 0.882 ms`, `p95 = 1.254 ms`, `max = 1.577 ms`, `ns3_exit = 0`.
3. **Otimização algorítmica e benchmark empírico com buffer saturado (`B = 2000`, 3 slices × 54 ações, 200 repetições após 20 de warmup):**
   - [`KernelQuantileRegressorV2.predict_and_uncertainty`](../../../ns-o-ran-gym/src/nsoran/sqr_v2/controller_v2.py#L22-L38) computa a matriz de kernel `K` uma única vez por janela, com equivalência numérica verificada `< 1e-14` contra `v1` (`predict` + `uncertainty`) em `n_mem ∈ {0, 1, 50, 2000}` e decisões idênticas em `test_predict_and_uncertainty_numerical_equivalence_and_speed`.
   - Latência observada em CPU do passo completo de 3 slices com `B = 2000` (`task-3334`):

| Operação (`B = 2000`, 3 slices) | `v1` separado (`p50 / p95 / max`) | `v2` passe único (`p50 / p95 / max`) | Speedup (`p50` / `mean`) | Folga vs `50 ms` |
|---|---|---|---|---|
| `score_candidates` (`3 × 54` avaliações) | `28.8050 / 30.5402 / 33.7835 ms` | `14.5128 / 15.3729 / 17.7462 ms` | **`1.985×` / `1.978×`** | ~32,3 ms no `max` |
| **`full act()` (3 slices completo)** | `43.2946 / 44.8501 / 50.1857 ms` | **`28.7299 / 30.2414 / 37.8475 ms`** | **`1.507×` / `1.525×`** | **`+12,15 ms` no `max`** (`v1` chegou a `50.19 ms`) |

---

## 5. Correção e Reanálise Somente-Leitura do `BUG-04` (`n = 40` runs `v1`)

O script somente-leitura [`sqr_v2_reanalyze_v1.py`](../../../ns-o-ran-gym/examples/sqr_v2_reanalyze_v1.py) usando [`campaign_analysis_v2.py`](../../../ns-o-ran-gym/src/nsoran/sqr_v2/campaign_analysis_v2.py) reanalisou os 40 runs originais em `results/sqr_campaign/` **sem modificar nenhum dado bruto**, gerando [`errata_bug04_v1_vs_v2.csv`](../../../ns-o-ran-gym/results/sqr_v2/reanalysis_v1_bug04/tables/errata_bug04_v1_vs_v2.csv), [`per_seed_metrics_bug04_fixed.csv`](../../../ns-o-ran-gym/results/sqr_v2/reanalysis_v1_bug04/tables/per_seed_metrics_bug04_fixed.csv) e gráficos corrigidos em [`results/sqr_v2/reanalysis_v1_bug04/figures/`](../../../ns-o-ran-gym/results/sqr_v2/reanalysis_v1_bug04/figures/).

Principais achados:
1. **Remoção do fator espúrio `3×` em `prb_idle_est`:** `rb_cell_total_idle_est_mean` passa a usar `.first()` por passo (`step`) antes da média temporal, reduzindo os valores em exatamente `3×` e restaurando a conservação por slot `rb_slot_executed_cell + rb_slot_unallocated_quota = C = 53.0` com erro **`0.0`** em todos os 40 runs.
2. **Separação estrita de unidades:** `PRBs/slot` (cota `[0, 53]`) vs `RBG·slots / janela 100 ms` (`scheduled_rbg`, `idle_rbg`, `capacity_rbg_slots = C × dl_slots`). Em `equal`/`kbrl`/`sqr-g0`/`sqr-g1`, os valores reportados de violação de SLA (`ue_time_violation_pct` e `slice_time_violation_pct`) **não se alteram** (permanecem ~`84.3%–89.9%`), porque a correção atinge apenas a agregação de ocupação/ociosidade de PRBs.

| Método (`10` seeds cada) | `v1` `idle_est` (soma `3×`) | `v2` `idle_est` (`.first()` / slot) | `v2` executado (`PRBs/slot`) | Capacidade `C` | Erro slot `|exec + idle - C|` | `v1` misto `cell_total_sched` | `v2` agendado/slot ativo | `v2` janela `sched / idle / cap` (`RBG·slots`) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `equal` | `0.000500` | `0.000167` | `52.999833` | `53.0` | `0.0` | `980.307690` | `22.158518` | `980.26 / 1539.75 / 2513.94` |
| `kbrl` | `0.194962` | `0.064987` | `52.935013` | `53.0` | `0.0` | `901.732811` | `20.375099` | `901.72 / 1635.04 / 2534.26` |
| `sqr-g0` | `74.958515` | **`24.986172`** | `28.013828` | `53.0` | `0.0` | `630.946789` | `14.294922` | `630.82 / 1898.76 / 2528.24` |
| `sqr-g1` | `76.746756` | **`25.582252`** | `27.417748` | `53.0` | `0.0` | `615.953962` | `14.075269` | `615.81 / 1909.00 / 2522.71` |

---

## 6. Síntese das Verificações Pós-Edição Final

- `bash MODA/ns-o-ran-gym/tests/run_sqr_v2_tests.sh -v`: **`8 passed in 139.25s`** (`0` falhas, `0` skips).
- `bash MODA/ns-o-ran-gym/tests/run_sqr_tests.sh -x -q`: **`360 passed, 2 warnings in 56.93s`** (`0` falhas).
- `sqr_v2_reanalyze_v1.py`: **`n_runs=40`**, saída completa em [`results/sqr_v2/reanalysis_v1_bug04`](../../../ns-o-ran-gym/results/sqr_v2/reanalysis_v1_bug04/).
- `verify_b0_frozen_manifest`: **`17/17` código/binário + `5/5` checkpoints** (`all_ok: true`).
- `Auditoria.txt` SHA-256: `cfe517c6efc96691d2630d9b72526eb681b460b95e8772b90c9acd5ebe177924` (inalterado).
- `origin/main`: `060d0581f950d041fb5c4f6482259528089cf019` (inalterado).

---

## 7. Limitações Identificadas Antes da Etapa 3A

1. **Discrepância de telemetria multi-beam no MAC scheduler legado (`sqr-mac-scheduler.cc:361,369,758-759`):**
   - `m_windowDlSlots` é incrementado uma única vez por chamada de slot em `AssignDLRBG`, enquanto `m_windowScheduledRbg` e `m_windowBudgetRbg` acumulam dentro do loop `for (const auto& el : activeDl)` (uma vez **por beam ativo no slot**).
   - Em todos os 40 runs do `v1`, janelas *single-beam* têm conservação exata `rb_win_singlebeam_conservation_abs_err = 0.0` e `rb_win_scheduled_le_budget_violations = 0.0` em 100% dos runs. Contudo, em janelas *multi-beam* (`rb_win_multibeam_budget_gt_cap_windows > 0`), o somatório acumulado por beam pode superar `total_rbg * dl_slots` (levando `prb_idle_rbg` a saturar em `0` e produzindo `v2_win_conservation_err` médio de `227.2` a `878.9` `RBG·slots` na errata acima).
   - Mantido intocado na Etapa 2 para preservar fidelidade `A/A` com o scheduler da `v1`; recomendado corrigir/expor contador explícito por beam/slot na **Etapa 3A** (isolado da validação IPC da Etapa 2).
2. **Escopo físico e aprendizado deliberadamente inalterados nesta Etapa 2:**
   - Em `sqr-v2-sim.cc`, PHY/MAC/MCS/TDD/QoS seguem idênticos ao `v1` (`P0_legacy_tdd7_uma_mcs_v1`) para isolar o efeito do `IPC Handshake v2` e da agregação `BUG-04`.
   - **`BUG-01` (MCS)** e **`BUG-06` (sentinela SNR)** aguardam autorização para a **Etapa 3A** (testes isolados sem incorporação automática ao baseline).
   - **Etapa 3B (TDD / QoS 5QI)** permanece **BLOQUEADA**, sujeita à revisão da contabilização dos símbolos DL no slot especial (`5G-LENA` vs `TS 38.331`) e à reavaliação de QoS a partir de `Q0`/`5QI=9`.
3. **Observação de versionamento git da Etapa 2:**
   - `MODA/ns-3-dev/scratch/.gitignore` ignora `scratch/*` por padrão (`/*`, com exceção `!sqr/`), razão pela qual `MODA/ns-3-dev/scratch/sqr_v2/` ainda não aparece listado em `git status` sem `!sqr_v2/`. Os demais arquivos Python, testes, documentos e saídas de reanálise estão como não rastreados na branch `exp/sqr-v2-infra`, aguardando decisão sobre commit local.
