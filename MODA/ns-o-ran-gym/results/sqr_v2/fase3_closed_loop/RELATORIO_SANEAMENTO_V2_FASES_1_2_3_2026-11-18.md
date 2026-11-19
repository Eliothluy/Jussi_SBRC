# Relatório Técnico de Saneamento, Migração Normativa e Validação em Malha Fechada (`v2`) — Fases 1, 2 e 3

**Data:** 2026-11-18  
**Branch / Commit base:** `exp/sqr-v2-infra` (`3f49436`, derivada de `168c503` / tag `v1.0-b0-frozen-audit-2026-11-17`)  
**Objetivo:** Eliminar os artefatos de implementação apontados em [`RELATORIO_CONSOLIDADO_LIMITACOES_NS3_3GPP_PESQUISA_2026-11-18.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/RELATORIO_CONSOLIDADO_LIMITACOES_NS3_3GPP_PESQUISA_2026-11-18.md), migrar para o padrão TDD 3GPP TS 38.331 de `5 ms` (`80` slots DL+S / `100 ms`), otimizar e calibrar o controlador `SQR-CER` em `dev`, e executar a avaliação completa em malha fechada (`Fase B`) preservando integralmente o baseline histórico `B0`/`v1` e o regime físico calibrado (Guardrail 1).

---

## 1. Resumo dos Veredictos por Fase

| Fase | Escopo | Veredicto | Evidência Principal |
|---|---|---|---|
| **Guardrail 1 & Integridade `B0`/`v1`** | Preservação do regime físico (`NrLteMiErrorModel`, `1x1 SISO`, `30 dBm`, `10 MHz`, `C=53`, `5QI=9`, `300 m`) e congelamento histórico | **PASS** | `verify_b0_frozen_manifest()`: `17/17` arquivos de código + `5/5` checkpoints (`all_ok: True`); suíte `v1`: `360/360 passed`; binário `v1` inalterado (`6403e2eb...`). |
| **FASE 1 (`Etapa 3A + Patch 1`)** | Remoção de `BUG-01`, correção multi-beam `BUG-08`, correção KPM/ciclo de vida `BUG-06` e `BUG-07`, restauração do `Patch 1` (`nr-pdcp.cc`) e preservação do `Patch 4` | **PASS** | `rb_win_conservation_abs_err = 0.0` em `100%` dos runs (`24/24` Fase 2, `105/105` Fase B, `30/30` sweep de raio); `0` overrides `MCS=15`, `0` sentinelas `-999`, `0` sessões fantasma `< 0.6 s`; `22/22 passed` na suíte `v2`. |
| **FASE 2 (`Etapa 3B`)** | Migração TDD `7 ms` (`"DL|DL|DL|DL|S|UL|UL|"`) → `5 ms` (`"DL|DL|DL|S|UL|"`), manutenção documentada de `5QI=9`, sweep URLLC $a_u \in \{5,10,15,20,25,30\}$ (`24` runs) | **PASS** | `dl_slots_max = 80` (`10 % 5 == 0`, `0` aborts `AssignCsiRsOffset`); taxa de empate em `p95` caiu de `69.4%` para `34.4975%`; `spearman_within_window_mean = -0.3666`; queda de `p95` de `92.79 ms` (`a_u=5`) para `9.71–10.30 ms` (`a_u=15..30`). |
| **FASE 3 (`Etapas 4, 5` & `Fase B`)** | Aceleração NumPy exata do kernel (`< 1 ms`), calibração `L3`/`U_preq` em `dev`, correções de blackout URLLC (`priority_hoashi` & vacuidade) e piso físico mMTC (`a_min_m=4`), campanha `105` episódios (`MAIN` + `OOS`) + `30` runs multi-raio | **PASS (infra & `MAIN`) / RESSALVA CIENTÍFICA (`OOS`)** | Infraestrutura `100%` limpa (`135/135` runs `exit 0`, `0` timeouts/gaps/erros de conservação; `cer_ms p95 = 0.359 ms`). No `MAIN` (`10` seeds), `sqr_cer` reduz violação conjunta para `45.14%` (`+11.257 pp` vs `sqr_g1` `56.27%`, win ratio `0.7718`). No `OOS` (`5` seeds), `sqr_cer` marca `49.04%` (`-2.099 pp` vs `sqr_g1` `46.94%`, vs `equal` `42.31%`), com win ratio condicional `0.5417` e vitória em `3/5` seeds (`3103`, `3104`, `3105`). |

---

## 2. Portão de Integridade, Verificação de Testes e `git diff --stat`

### 2.1 Verificação Final Automatizada (após todas as edições)

1. **Manifesto congelado `B0`:**
   ```text
   verify_b0_frozen_manifest() -> {'code_total': 17, 'code_matched': 17, 'code_mismatches': [],
                                   'checkpoints_total': 5, 'checkpoints_matched': 5, 'checkpoint_mismatches': [],
                                   'all_ok': True}
   ```
   Hash SHA-256 dos binários `ns-3`:
   - `v1` preservado: `6403e2ebdc7a7318adf18f37acb674134a9fc412421c3cbb0dd1d5528c9eb0b1` ([`ns3.46-sqr-sim-default`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/build/scratch/sqr/ns3.46-sqr-sim-default))
   - `v2` compilado/validado: `8effcaebb9183bc8116a39bf071d60ce1de9e4757aec60fa1dfcb048b52c40f9` ([`ns3.46-sqr-v2-sim-default`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/build/scratch/sqr_v2/ns3.46-sqr-v2-sim-default))
2. **Suíte histórica `v1` (`bash MODA/ns-o-ran-gym/tests/run_sqr_tests.sh`):**
   - Resultado: **`360 passed, 2 warnings in 59.26s`** (`Command exited with code 0`).
3. **Suíte `v2` completa (`bash MODA/ns-o-ran-gym/tests/run_sqr_v2_tests.sh -v`):**
   - Resultado: **`22 passed in 173.54s (0:02:53)`** (`Command exited with code 0`).
   - Inclui testes unitários e com `ns-3` real: governança de manifesto/seeds, equivalência numérica (`0-ULP`) e velocidade do kernel, sincronização IPC `v2` e injeção de 4 classes de falhas + atrasos, paridade A/A estrita, conservação multi-seed/banda, 8 testes de `CER`/`PriorityHoashi`/`PAVA`/`A2A`, e 6 testes em [`test_sqr_v2_etapa3_fixes.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/tests/test_sqr_v2_etapa3_fixes.py) (`BUG-01` + `Patch 1`, `BUG-06`, sanidade `ns-3` real de conservação e ausência de fantasmas/`-999`, kernel anisotrópico `L3`, smoke `sqr_cer` em `tdd5ms`, e não-vacuidade sob blackout URLLC).

### 2.2 Inventário de Alterações (`git diff --stat` e Árvore C++ `ns-3-dev`)

**Arquivos Python/Shell modificados e rastreados pelo Git (`git diff --stat` vs `3f49436`, idêntico na raiz e em `MODA`):**

```text
 MODA/ns-o-ran-gym/examples/sqr_v2_run.py           | 944 +++++++++++++++++----
 MODA/ns-o-ran-gym/src/nsoran/sqr_v2/__init__.py    |  36 +-
 MODA/ns-o-ran-gym/src/nsoran/sqr_v2/controller_v2.py | 219 ++++-
 MODA/ns-o-ran-gym/src/nsoran/sqr_v2/env_v2.py      | 104 ++-
 MODA/ns-o-ran-gym/tests/run_sqr_v2_tests.sh        |   2 +-
 5 files changed, 1145 insertions(+), 160 deletions(-)
```

**Novos arquivos Python (não rastreados até o commit):**
- [`cer_reallocator.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_v2/cer_reallocator.py) (`989` linhas) e [`priority_hoashi.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_v2/priority_hoashi.py)
- [`sqr_v2_cer_offline_check.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_v2_cer_offline_check.py), [`sqr_v2_tdd5ms_urllc_sweep.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_v2_tdd5ms_urllc_sweep.py) (`335` linhas), [`sqr_v2_fase3_calibrate_l3_upreq.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_v2_fase3_calibrate_l3_upreq.py) (`623` linhas), [`sqr_v2_fase3_closed_loop_eval.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_v2_fase3_closed_loop_eval.py) (`467` linhas)
- [`test_sqr_v2_cer.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/tests/test_sqr_v2_cer.py) e [`test_sqr_v2_etapa3_fixes.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/tests/test_sqr_v2_etapa3_fixes.py) (`302` linhas)

**Arquivos C++ modificados (diretórios sob `.gitignore` em `ns-3-dev/scratch/.gitignore` e `ns-3-dev/contrib/.gitignore`, fora do `git diff --stat`):**
- [`nr-pdcp.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/contrib/nr/model/nr-pdcp.cc) (`245` linhas) — `Patch 1` restaurado (`L207+`); `Patch 4` mantido em [`nr-qos-rule-classifier.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/contrib/nr/model/nr-qos-rule-classifier.cc).
- [`rslaq-mac-scheduler.h`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/rslaq-mac-scheduler.h) (`406` linhas), [`rslaq-mac-scheduler.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/rslaq-mac-scheduler.cc) (`1636` linhas), [`sqr-mac-scheduler.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-mac-scheduler.cc) (`945` linhas) e [`sqr-v2-sim.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc) (`3409` linhas).

---

## 3. FASE 1 — Saneamento do Simulador (`Etapa 3A + Patch 1`)

| Bug / Patch | Localização Exata | Correção Implementada | Validação |
|---|---|---|---|
| **`BUG-01` (MCS=15 forçado)** | [`rslaq-mac-scheduler.cc:1551`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/rslaq-mac-scheduler.cc#L1551) | Removido o override `if (ueInfo->m_dlMcs == 0) ueInfo->m_dlMcs = 15;`, preservando `m_dlMcs == 0` da adaptação de enlace (AMC). | `test_bug01_forced_mcs15_override_absent_and_patch1_present` (`PASSED`). |
| **`BUG-08` (Conservação multi-beam)** | [`rslaq-mac-scheduler.h:366-368`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/rslaq-mac-scheduler.h#L366-L368), [`sqr-mac-scheduler.cc:42-85`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-mac-scheduler.cc#L42-L85), [`869-881`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-mac-scheduler.cc#L869-L881), [`908-919`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-mac-scheduler.cc#L908-L919) | Introduzidos acumuladores ponderados pela fração de símbolos OFDM do beam (`beamSym / symAvail`) e materialização inteira de fim de slot via método dos maiores restos (`MaterializeConservedWindowCounts`), garantindo `sum(sched) <= sum(budget) <= C * m_windowDlSlots`. | `rb_win_conservation_abs_err = 0.0` em **todos** os `159` runs realizados após a correção (contra erros históricos de `227–879 RBG·slots`). |
| **`BUG-06` (`-999` em CQI/SINR)** | [`rslaq-mac-scheduler.cc:1621-1634`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/rslaq-mac-scheduler.cc#L1621-L1634), [`sqr-mac-scheduler.cc:536`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-mac-scheduler.cc#L536), [`sqr-v2-sim.cc:2131`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc#L2131), [`env_v2.py:106`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_v2/env_v2.py#L106) | `GetUeDlSinrDb` retorna `quiet_NaN()` quando não há reporte válido (`-1`) ou `wbCqi == 0`; `sqr-v2-sim.cc` emite `"nan"` no KPM; `SqrV2KpmParser` mapeia explicitamente `wbCqi == 0` para piso de outage `0.0 dB` (`snr_was_imputed = False`). | `test_bug06_v2_kpm_parser_out_of_range_and_missing_cqi` & `test_etapa3a_real_ns3_sanity_conservation_and_no_ghosts_or_999` (`PASSED`). |
| **`BUG-07` (Sessões transientes & UEs fantasmas)** | [`sqr-v2-sim.cc:312`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc#L312), [`1061-1075`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc#L1061-L1075), [`1147-1153`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc#L1147-L1153), [`2038-2056`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc#L2038-L2056) | Guarda `ue.attached` na coleta de KPIs/KPMs, filtro de sessões transientes `< g_attachGraceSec + 0.1 s` (`< 0.6 s`) sem iniciar `clientApp`, e `ue.clientApp->Dispose()` em `UeDeparture`. | `test_etapa3a_real_ns3_sanity_conservation_and_no_ghosts_or_999` (`PASSED`). |
| **`Patch 1` & `Patch 4` + Guardrail 1** | [`nr-pdcp.cc:207+`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/contrib/nr/model/nr-pdcp.cc#L207), [`sqr-v2-sim.cc:3167-3177`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc#L3167-L3177) | `Patch 1` restaurado (idêntico em código ao backup [`MODA/docs/patches_nr/nr-pdcp.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/patches_nr/nr-pdcp.cc)), `Patch 4` preservado, premissas PHY/QoS do Guardrail 1 documentadas em comentário explícito. | `0` aborts `NS_ASSERT` em `159/159` simulações pós-correção. |

---

## 4. FASE 2 — Migração Normativa TDD (`7 ms -> 5 ms`) e Sweep URLLC (`Etapa 3B`)

- Alteração padrão em [`sqr-v2-sim.cc:2743-2744`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc#L2743-L2744): `tddPattern = "DL|DL|DL|S|UL|"` (`5 ms` TS 38.331, `80` slots DL+S por janela de `100 ms`, divisor exato de `10 ms` para `AssignCsiRsOffset`), com `5QI = 9` mantido e documentado.
- Campanha de sweep ([`fase2_sweep_summary.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase2_tdd5ms_urllc_sweep/fase2_sweep_summary.json) e [`urllc_sweep_by_action.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase2_tdd5ms_urllc_sweep/urllc_sweep_by_action.csv)): `24` runs (`2` seeds dev `{101, 102}` × `6` ações `a_u ∈ {5, 10, 15, 20, 25, 30}` × `2` modos TDD, `16.0 s`, `sliceLambdas = 12,36,12`, raio `300 m`).

| Modo TDD | `dl_slots_max` | `tie_rate_pct` | `improved / worsened` | `mean_dy_ms` | `spearman_pooled` | `spearman_within_win` | `cons_err_max` |
|---|---|---|---|---|---|---|---|
| `tdd5ms_default` (`DL\|DL\|DL\|S\|UL\|`) | `80` | `34.4975%` (`484/1403`) | `550 / 369` | `-13.5075 ms` | `-0.1274` (`p=4.89e-06`) | `-0.3666` | `0.0` |
| `tdd7ms_legacy` (`DL\|DL\|DL\|DL\|S\|UL\|UL\|`) | `58` | `32.6409%` (`440/1348`) | — | — | — | `-0.4469` | `0.0` |

**Curva dose-resposta URLLC (`p95_mean ms` / `miss_pct %` em `a_u`):**

| `a_u` (PRBs) | `tdd5ms_default`: URLLC `p95` (`miss %`) | `tdd7ms_legacy`: URLLC `p95` (`miss %`) |
|---|---|---|
| `5` | `92.79 ms` (`40.13%`) | `139.39 ms` (`59.37%`) |
| `10` | `35.98 ms` (`30.25%`) | `92.38 ms` (`37.33%`) |
| `15` | `9.80 ms` (`7.53%`) | `14.15 ms` (`23.47%`) |
| `20` | `10.30 ms` (`10.77%`) | `11.15 ms` (`13.24%`) |
| `25` | `10.06 ms` (`8.34%`) | `11.68 ms` (`14.00%`) |
| `30` | `9.71 ms` (`7.71%`) | `11.77 ms` (`13.27%`) |

---

## 5. FASE 3 — Otimização, Calibração (`dev`) e Campanha em Malha Fechada (`Fase B`)

### 5.1 Otimização de Inferência em Tempo Real (`controller_v2.py` e `cer_reallocator.py`)
- [`controller_v2.py:41-90`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_v2/controller_v2.py#L41-L90): pré-computação em NumPy (`da2_pre`, `buf`, `buf2`, `dz_buf`, `np.einsum("ij,ij->i", ...)`) e `warm_candidate_cache` (`165-180`) preservando equivalência numérica de **`0-ULP`** contra o caminho de referência `v1` (`c1.act(z) == c2.act(z)`).
- [`cer_reallocator.py:93, 424, 726`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_v2/cer_reallocator.py#L93): adição de `dl_slots_per_win` em `CerConfig`, propagado dinamicamente para `compute_mmtc_physical_floor` (`80` em `tdd5ms`).
- **Latência verificada:** em execução isolada (`40 s`, `3,3,3`, `SQR-CER`), `agent_decision_ms p50 = 0.757 ms`, `p95 = 0.940 ms` (pós-warmup `0.795 ms / 0.956 ms`) e `cer_reallocate_ms p50 = 0.145 ms`, `p95 = 0.179 ms`, com `0` holds. Sob contenção de `10` workers paralelos na `Fase B`, `cer_ms` permaneceu em **`p50 = 0.2317 ms / p95 = 0.3594 ms`** (`MAIN`) e **`0.2240 ms / 0.3011 ms`** (`OOS`), e o ciclo completo `agent_ms` em **`1.3526 ms / 2.3212 ms`** (`MAIN`).

### 5.2 Calibração `dev` (`{101, 102, 103}`): `FeatureScaler`, `L3` ($\ell_a$) e `U_preq`
- Conjunto pesado de estresse (`5,5,5`, `120 s`, `7.188` amostras) arquivado em [`fase3_calibration_heavy_555/`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_calibration_heavy_555/calibration_summary.json).
- Calibração ativa (`3,3,3`, `90 s`, `5.388` amostras) consolidada em [`calibration_summary.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_calibration/calibration_summary.json):
  - [`scaler_manifest_v2_tdd5ms.yaml`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_calibration/scaler_manifest_v2_tdd5ms.yaml): limites `p1–p99` atualizados para `tdd5ms` (`n_u_p99: 3.0 -> 8.0`, `queue_u_p99: 709.0 -> 6405.0`, `offered_e_p99: 16.5024 -> 22.7366`, `offered_u_p99: 1.4256 -> 3.811824`, `queue_e_p99: 9844456.46 -> 7305094.62`, `queue_m_p99: 1422.0 -> 711.0`).
  - **`L3` anisotrópico selecionado (`selected_action_length_scales = [0.35, 0.2, 0.1]`):** em [`l3_calibration_sweep.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_calibration/l3_calibration_sweep.csv), `eMBB` caiu `flattening_d8` de `100.0%` (`ell_a=1.0`) para `63.07%` (`ell_a=0.35`, `l3_score=1.0212`); `URLLC` reduziu `flattening_d8` de `80.49%` para `50.57%` e elevou `spearman_f_vs_m` de `0.1956` para `0.6343` (`ell_a=0.2`, `l3_score=0.9194`); para `mMTC`, selecionou-se `ell_a=0.1` (`l3_score=0.8463`, `spearman=0.8648`, privilegiando sensibilidade local frente a `ell_a=1.0` `l3_score=0.9289` com `100.0%` de flattening).
  - **`U_preq` (`beta_v=0.85, beta_o=0.15, beta_c=0.15`) vs `sigma_kernel`** ([`upreq_calibration_grid.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_calibration/upreq_calibration_grid.csv)): AUROC superior nos 3 slices — `eMBB`: **`0.9260`** vs `0.8378`; `URLLC`: **`0.9358`** vs `0.5169`; `mMTC`: **`0.9888`** vs `0.2523`.
  - **Emenda de piso físico em [`cer_config_v2_calibrated.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_calibration/cer_config_v2_calibrated.json):** `dl_slots_per_win = 80`, `a_min_m = 4`, `theta_u_def_e = 0.274`, `theta_u_def_u = 0.38`, `theta_u_don_e = 0.15`, `theta_u_don_u = 0.2`, `theta_u_don_m = 0.15` (ajustado de `0.30` após pico de cauda em `seed=3009`).

### 5.3 Descobertas de Auditoria Crítica & Correções Aplicadas Antes da Execução Final
1. **Correção do `PriorityHoashiReallocator` e remoção de vacuidade sob blackout URLLC (`sqr_v2_run.py`):**
   - Antes da correção (execução arquivada em [`fase3_closed_loop_pre_blackout_fix/`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop_pre_blackout_fix/)), `PriorityHoashiReallocator.allocate` recebia `d.raw_action` em vez de `d.action`, produzindo `a_exec_1 = 0` em `98/98` janelas na `seed=3002` (`8.724/8.724` pacotes URLLC perdidos), e a ausência de entregas (`DelayStream.stats == None`) era pontuada como `0.0%` de violação.
   - Após a correção (`d.action` passado ao realocador + regra em `_summarize_and_write_het_windows` e no loop online: `urllc_tx_pkts > 0` com `0` entregas → `urllc_miss_pct = 100.0`, `margin_u = -1.0`, `sla_ok_u = 0`, `sla_ok_all3 = 0`; somente `urllc_tx_pkts == 0` é vácuo `sla_ok_u = 1`), testado em `test_urllc_blackout_not_vacuous_when_tx_pkts_positive`, a `seed=3002` de `priority_sqr` passou a ter `a_exec_1` médio de `20.73 PRBs` e `88/98` janelas válidas (`p95 = 8.713 ms`).
2. **Causa raiz física do `mmtc_pdr = 0.0` em `PropDemandController`:**
   - Sem `BUG-01`, `Prop` fixa piso de `5%` (`3 PRBs`) para mMTC. Com `m_dlMcs = 0` inicial, `CreateDlDci` ([`nr-mac-scheduler-ofdma.cc:770`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/contrib/nr/model/nr-mac-scheduler-ofdma.cc#L770): `if (tbs < 10) return nullptr;`) descarta a emissão de DCI DL para `3 PRBs`, travando o mMTC em `MCS = 0` (`tx_pkts=24, rx_pkts=0`, `74.35–80.49%` de violação mMTC). A partir de `4 PRBs` (`a_min_m = 4` no `cer_config_v2_calibrated.json`, `Equal`, `SQR`, `KBRL`), `tbs >= 10 B` e o PDR atinge `0.99–1.00`.

### 5.4 Resultados `Fase B`: `MAIN` (`3001..3010`) e `OOS` (`3101..3105`)

Fonte: [`fase3_method_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_method_summary.csv) e [`fase3_rescue_harm_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_rescue_harm_summary.csv) (`105` episódios, `90.0 s`, `warmup_s = 25.0 s`, `sliceLambdas = 3,3,3`; **todos** com `ns3_exitcode = 0`, `timeout_hold_steps = 0`, `window_gap_steps = 0`, `rb_win_conservation_abs_err = 0.0`).

| Split | Método | Violação Conjunta (%) | eMBB (%) | URLLC (%) | mMTC (%) | eMBB Thr (Mbps) | URLLC `p95` Mean (`p95` das janelas) | mMTC PDR | PRB Util (%) | `agent_ms` p50 / p95 | `cer_ms` p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **MAIN** | **`sqr_cer`** | **`45.14 ± 24.68`** | `28.94` | `21.99` | **`0.71`** | `7.02` | `11.60` (`32.82`) ms | `1.014` | **`98.81%`** | `1.353 / 2.321` ms | `0.232 / 0.359` ms |
| **MAIN** | `sqr_g0` | `53.29 ± 17.96` | `32.56` | `29.31` | `4.10` | `6.31` | `12.37` (`22.60`) ms | `0.991` | `71.66%` | `1.579 / 2.876` ms | — |
| **MAIN** | `sqr_g1` | `56.27 ± 21.24` | `36.40` | `31.82` | `3.77` | `6.49` | `19.15` (`44.98`) ms | `0.992` | `70.60%` | `1.680 / 2.971` ms | — |
| **MAIN** | `priority_sqr` | `56.99 ± 22.24` | `51.34` | **`7.16`** | `28.44` | `3.96` | **`7.63`** (`11.45`) ms | `0.679` | `92.40%` | `1.384 / 2.517` ms | `0.019 / 0.026` ms |
| **MAIN** | `equal` | `60.79 ± 27.26` | `48.52` | `23.84` | `0.43` | `5.08` | `8.91` (`13.21`) ms | `0.999` | `100.00%` | `0.048 / 0.060` ms | — |
| **MAIN** | `kbrl` | `63.19 ± 26.05` | `50.63` | `26.73` | `2.95` | `3.58` | `9.98` (`14.08`) ms | `0.997` | `98.46%` | `1.974 / 3.115` ms | — |
| **MAIN** | `prop` | `82.70 ± 20.17` | `18.21` | `22.93` | `74.35` | `9.62` | `9.67` (`14.47`) ms | `0.000` | `98.46%` | `0.155 / 0.205` ms | — |
| **OOS** | `equal` | **`42.31 ± 34.25`** | `35.46` | `33.77` | **`0.06`** | `5.35` | `10.01` (`13.89`) ms | `1.000` | `100.00%` | `0.047 / 0.060` ms | — |
| **OOS** | `sqr_g1` | `46.94 ± 28.55` | `23.36` | `38.98` | `3.02` | `5.94` | `17.50` (`54.77`) ms | `0.987` | `68.42%` | `1.499 / 2.882` ms | — |
| **OOS** | **`sqr_cer`** | `49.04 ± 28.20` | `26.79` | `38.36` | `0.56` | `6.76` | `20.10` (`55.98`) ms | `0.998` | **`100.00%`** | `1.287 / 2.245` ms | `0.224 / 0.301` ms |
| **OOS** | `sqr_g0` | `50.68 ± 30.08` | `28.80` | `43.58` | `2.62` | `7.38` | `15.72` (`33.65`) ms | `0.987` | `74.82%` | `1.524 / 2.860` ms | — |
| **OOS** | `priority_sqr` | `59.63 ± 26.25` | `51.33` | **`30.09`** | `19.81` | `3.22` | **`9.69`** (`14.47`) ms | `0.738` | `81.27%` | `1.142 / 2.593` ms | `0.016 / 0.026` ms |
| **OOS** | `kbrl` | `62.62 ± 26.30` | `54.26` | `36.79` | `2.59` | `3.37` | `13.55` (`25.45`) ms | `0.995` | `98.39%` | `1.692 / 2.941` ms | — |
| **OOS** | `prop` | `96.70 ± 3.24` | `7.35` | `47.28` | `80.49` | `10.60` | `12.46` (`17.50`) ms | `0.000` | `92.83%` | `0.145 / 0.199` ms | — |

**Análise Pareada `RESCUE vs HARM` contra `sqr_g1`:**

| Split | Método | Janelas Pareadas | RESCUE (%) | HARM (%) | Ganho Líquido (pp) | Janelas Intervencionadas | RESCUE / HARM (interv.) | Win Ratio Condicional |
|---|---|---|---|---|---|---|---|---|
| **MAIN** | **`sqr_cer`** | `6405` | **`20.328%`** (`1302`) | **`9.071%`** (`581`) | **`+11.257 pp`** | `2956` | `744 / 220` | **`0.7718`** |
| **MAIN** | `sqr_g0` | `6405` | `14.317%` (`917`) | `11.304%` (`724`) | `+3.013 pp` | `6405` | `917 / 724` | `0.5588` |
| **MAIN** | `priority_sqr` | `6405` | `10.476%` (`671`) | `11.210%` (`718`) | `-0.734 pp` | `6405` | `671 / 718` | `0.4831` |
| **MAIN** | `equal` | `6405` | `13.973%` (`895`) | `18.548%` (`1188`) | `-4.575 pp` | `6405` | `895 / 1188` | `0.4297` |
| **MAIN** | `kbrl` | `6405` | `19.625%` (`1257`) | `26.635%` (`1706`) | `-7.010 pp` | `6405` | `1257 / 1706` | `0.4242` |
| **MAIN** | `prop` | `6405` | `5.074%` (`325`) | `31.819%` (`2038`) | `-26.745 pp` | `6405` | `325 / 2038` | `0.1375` |
| **OOS** | `equal` | `3240` | `11.975%` (`388`) | `7.346%` (`238`) | `+4.630 pp` | `3240` | `388 / 238` | `0.6198` |
| **OOS** | **`sqr_cer`** | `3240` | `7.500%` (`243`) | `9.599%` (`311`) | **`-2.099 pp`** | `1092` | `156 / 132` | **`0.5417`** |
| **OOS** | `sqr_g0` | `3240` | `3.488%` (`113`) | `7.222%` (`234`) | `-3.735 pp` | `3240` | `113 / 234` | `0.3256` |
| **OOS** | `priority_sqr` | `3240` | `3.148%` (`102`) | `15.833%` (`513`) | `-12.685 pp` | `3240` | `102 / 513` | `0.1659` |
| **OOS** | `kbrl` | `3240` | `10.216%` (`331`) | `25.895%` (`839`) | `-15.679 pp` | `3240` | `331 / 839` | `0.2829` |
| **OOS** | `prop` | `3240` | `0.957%` (`31`) | `50.710%` (`1643`) | `-49.753 pp` | `3240` | `31 / 1643` | `0.0185` |

*Diagnóstico por seed no `OOS` (`fase3_per_seed_summary.csv`):* `sqr_cer` supera `sqr_g1` em `3/5` seeds (`3103`: `42.28% vs 46.30%`, `3104`: `16.20% vs 18.36%`, `3105`: `93.67% vs 94.44%`), mas sofre perda de URLLC na `3101` (`51.70% vs 34.57%`) e leve perda na `3102` (`41.36% vs 41.05%`), além de cauda `urllc_p95_ms_p95` elevada sob saturação extrema em `seed=3009` (`203.78 ms`) e `seed=3105` (`211.70 ms`).

### 5.5 Sweep Multi-Raio Sanitizado (`dev {101, 102, 103}`, `30` runs)

Fonte: [`fase3_radius_sweep_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_radius_sweep_summary.csv). Todos com `cons_err = 0.0` e `0` aborts.

| Raio (`m`) | `embb_article_05_10_15`: Viol (%) | `thr_p5` (Mbps) | PRB Util (%) | `het_sqr_cer`: Viol (%) | `thr_p5` (Mbps) | URLLC `p95` (ms) | PRB Util (%) |
|---|---|---|---|---|---|---|---|
| `300` | `44.59%` | `1.873` | `39.95%` | `46.87%` | `6.857` | `12.38 ms` | `100.0%` |
| `600` | `52.24%` | `1.439` | `43.40%` | `50.34%` | `5.144` | `20.34 ms` | `100.0%` |
| `1000` | `61.68%` | `1.159` | `44.90%` | `55.59%` | `3.174` | `77.62 ms` | `100.0%` |
| `1500` | `68.80%` | `0.923` | `53.60%` | `80.76%` | `1.254` | `135.75 ms` | `100.0%` |
| `2000` | `78.64%` | `0.771` | `59.27%` | `83.67%` | `0.586` | `389.15 ms` | `100.0%` |

Confirma-se fisicamente que a degradação do artigo a `2000 m / 30 dBm / 1x1 SISO` permanece severa mesmo após a remoção completa de `BUG-01`/`BUG-06`/`BUG-07`/`BUG-08` e com `Patch 1` restaurado.
