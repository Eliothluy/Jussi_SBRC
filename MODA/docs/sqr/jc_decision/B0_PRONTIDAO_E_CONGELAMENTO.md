# Prontidão Final e Congelamento de `B0-rest` (`A1+A2a+A_scaler`)

**Data:** 2026-11-15  
**Decisão de Congelamento:** **GO — `B0-rest` CONGELADO (definitivo)** como configuração única `A1+A2a+A_scaler` (*SQR-HET adaptado e corrigido*), em [B0_FROZEN_MANIFEST.json](B0_FROZEN_MANIFEST.json).  
**Escopo respeitado:** Nenhuma alteração de parâmetros foi feita em `B0`; artefatos históricos `A0–A4` e `A_scaler` permanecem intactos; o contrato de congelamento/paridade causal está versionado para `D3-v2`; `A5`, `D3-v2` confirmatória, `PILOT`, `MAIN`, `A2b`, `A3b` e *seeds* `3101–3103` permanecem fora deste commit de fechamento de `B0-rest`.

---

## 1. Checagem de Robustez: `seed=2103` ($\lambda \in \{5, 6\}$, *cold start*) vs. `seed=2102`

Os dois runs em `seed=2103` (`MODA/ns-o-ran-gym/results/sqr_jc_decision/b0_robustness_s2103_v9_va0_guard_eum/het_lam{5,6}_s2103`) executaram com parâmetros e código idênticos ao candidato `A1+A2a+A_scaler`, sem erros funcionais ou de integridade:

### 1.1 Integridade, Execução e Segurança de Contrato

| Verificação | `s2102 λ=5` | `s2103 λ=5` | `s2102 λ=6` | `s2103 λ=6` |
|---|---:|---:|---:|---:|
| `ns3_exit_code == 0` / `kpm_max_ts_ms` | True / 299900 | True / 299900 | True / 299900 | True / 299900 |
| `kpm_missing_steps` / `rx == kpm` / `dups` | 0 / True / 0 | 0 / True / 0 | 0 / True / 0 | 0 / True / 0 |
| Decisões (`boot` / `het` / `closed`) | 2298 (303/1995/1994) | 2923 (975/1948/1947) | 2455 (460/1995/1994) | 2974 (979/1995/1994) |
| Erro reconstrução `z[0..1]` (`"eum"`) | ≤ 5e-7 | ≤ 5e-7 | ≤ 5e-7 | ≤ 5e-7 |
| Erro `z[4..5]` (`mean_field=False`) | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| `uTX` / `rx` / `unrx @ 300s` | 119182 / 119158 / **24** (0,020%) | 230686 / 230672 / **14** (0,006%) | 215464 / 215430 / **34** (0,016%) | 272272 / 272258 / **14** (0,005%) |

Ambos os checkpoints de `seed=2103` registraram o mesmo `config_sha256` (`a2b40861ace7ba87ce2d321ccc62f42f22e6b3798c4432171875ae55d7040d7c`) com verificação válida (`payload_sha256_valid=True`).

### 1.2 Desempenho, Projeções, SLAs R19 e Caudas URLLC

| Métrica | `s2102 λ=5` | `s2103 λ=5` (Δ vs 2102) | `s2102 λ=6` | `s2103 λ=6` (Δ vs 2102) |
|---|---:|---:|---:|---:|
| Utilização DL (`util_dl_pct`) | 46,75% | 41,19% (-5,56 pp) | 48,01% | 41,04% (-6,97 pp) |
| PRBs exec médios (`prb_exec_sum`) | 39,80 | 45,67 (+5,87) | 45,08 | 51,54 (+6,46) |
| Projeções `all` / `<3` / `=3` ativas | 42,78% / 17,15% / 82,08% | **54,13%** / 9,98% / 68,22% | 62,29% / 30,74% / 83,29% | **83,55%** / 51,50% / 90,78% |
| `eMBB` violação (`n_obs` / `unk`) | 11,92% (1519 / 475) | **30,74%** (1578 / 369) | 22,26% (1680 / 314) | **52,53%** (1759 / 235) |
| `URLLC R19` violação (`n_obs` / `unk`) | 17,51% (891 / 1103) | **41,94%** (1855 / 92) | 32,78% (1315 / 679) | **52,54%** (1989 / 5) |
| `mMTC` violação (`n_obs` / `unk`) | 1,18% (1944 / 50) | 1,27% (1652 / 295) | 1,33% (1962 / 32) | 1,59% (1763 / 231) |
| `STRICT_ALL3_R19` (`PASS`/`FAIL`/`UNK`) | **71,72%** (563/222/1209; unk 60,63%) | **44,48%** (657/820/470; unk 24,14%) | **55,30%** (662/535/797; unk 39,97%) | **28,76%** (465/1152/377; unk 18,91%) |
| `STRICT_DEMAND_R19` (`PASS`/`FAIL`/`UNK`) | 83,22% (1656/334/4) | 53,17% (1032/909/5) | 66,75% (1329/662/3) | 33,33% (663/1326/5) |
| `e∧u R19` (`PASS`/`FAIL`/`UNK`) | 569 / 216 / 1209 | 670 / 818 / 459 | 678 / 549 / 767 | 484 / 1271 / 239 |
| URLLC coorte `p95` / `p99` / `max` (ms) | 8,10 / 14,10 / 108,10 | 12,10 / 16,10 / 116,10 | 10,10 / 15,10 / 89,10 | 12,10 / 15,10 / 113,10 |
| URLLC pacotes `> 10 ms` (`uTX miss`) | 3,38% | **8,65%** | 5,50% | **10,12%** |

### 1.3 Calibração Quantílica, Ações e Modos (`seed=2103` vs `seed=2102`)

| Slice | Métrica | `s2102 λ=5` | `s2103 λ=5` | `s2102 λ=6` | `s2103 λ=6` |
|---|---|---|---|---|---|
| **eMBB (0)** | `tau_mean` (`tau_fin`) / `cov_exec` / `pb(tau)` / `pb(0.05)` | 0,0202 (0,010) / 2,30% / 0,0201 / 0,0454 | 0,0331 (0,010) / 4,82% / 0,0309 / 0,0394 | 0,0213 (0,010) / 2,50% / 0,0158 / 0,0302 | 0,0329 (0,010) / 4,72% / 0,0272 / 0,0355 |
| **eMBB (0)** | `a_raw` / `a_prop` / `a_exec` / `M_mean` | 26,57 / 26,95 / 23,78 / 40,11 | 24,94 / 25,48 / 20,21 / 40,36 | 29,79 / 30,30 / 23,60 / 40,99 | 26,10 / 26,41 / 19,10 / 21,31 |
| **eMBB (0)** | Modos (`conf` / `knn` / `soft`) | 1056 / 845 / 93 | 1407 / 483 / 57 | 1361 / 571 / 62 | 1074 / 835 / 85 |
| **URLLC (1)** | `tau_mean` (`tau_fin`) / `cov_exec` / `pb(tau)` / `pb(0.05)` | 0,0224 (0,010) / 2,81% / 0,0482 / 0,0600 | 0,0105 (0,010) / 1,67% / 0,0322 / 0,0589 | 0,0189 (0,010) / 2,74% / 0,0293 / 0,0442 | 0,0102 (0,010) / 1,01% / 0,0212 / 0,0552 |
| **URLLC (1)** | `a_raw` / `a_prop` / `a_exec` / `M_mean` | 12,12 / 12,31 / 9,89 / 47,47 | 24,41 / 24,73 / 18,90 / 50,00 | 12,37 / 12,58 / 9,14 / 45,60 | 33,26 / 33,66 / 24,18 / 50,00 |
| **URLLC (1)** | Modos (`conf` / `knn` / `soft`) | 174 / 1636 / 184 | **0 / 1754 / 193** | 16 / 1782 / 196 | **0 / 1795 / 199** |
| **mMTC (2)** | `tau_mean` (`tau_fin`) / `cov_exec` / `pb(tau)` / `pb(0.05)` | 0,0250 (0,015) / 1,29% / 0,0090 / 0,0104 | 0,0221 (0,020) / 1,45% / 0,0130 / 0,0155 | 0,0218 (0,010) / 1,33% / 0,0086 / 0,0111 | 0,0197 (0,015) / 1,59% / 0,0157 / 0,0222 |
| **mMTC (2)** | `a_raw` / `a_prop` / `a_exec` / `M_mean` | 6,54 / 6,69 / 6,09 / 44,98 | 9,55 / 9,68 / 7,07 / 35,36 | 16,12 / 16,41 / 12,33 / 45,18 | 12,02 / 12,26 / 8,26 / 42,21 |
| **mMTC (2)** | Modos (`conf` / `knn` / `soft`) | 1971 / 22 / 1 | 1694 / 227 / 26 | 1923 / 66 / 5 | 1218 / 700 / 76 |

**Diagnóstico da variação entre *seeds* (sem bug de código — ver auditoria completa em [`AUDITORIA_VARIABILIDADE_SEEDS_2102_2103.md`](AUDITORIA_VARIABILIDADE_SEEDS_2102_2103.md)):**  
Na `seed=2103`, o sorteio estocástico de chegadas/sessões produziu **muito mais atividade simultânea em URLLC** do que em `2102`: com configuração de geração idêntica (`20` vs `19` UEs em $\lambda=5$; `22` vs `22` em $\lambda=6$), duas sessões URLLC `Exp(60 s)` longas dobraram os UE·segundos URLLC (`153,2 s -> 332,6 s` em $\lambda=5$, com `891 -> 1855` janelas e `119.182 -> 230.686` TX no pós-warmup) e saturaram $\lambda=6$ (`1315 -> 1989`, `no_demand: 676 -> 0`, `uTX: 215.464 -> 272.272`), além de ampliar o *bootstrap* ativo (`303/460 -> 975/979`). Em $\lambda=5$, os `PASS` absolutos em `STRICT_ALL3_R19` **aumentaram (`563 -> 657`)**, mas a redução de `UNKNOWN` (`60,63% -> 24,14%`) quase dobrou o denominador condicional (`785 -> 1477`) sob competição trilateral (`39,47% -> 75,81%` em `=3` ativas), levando a taxa condicional de `71,72%` para `44,48%` (`28,76%` em $\lambda=6$). As referências exploratórias (`>65%` em $\lambda=5$ e `>45%` em $\lambda=6$) **não foram atingidas** e a robustez científica inter-seed permanece **não demonstrada**, sem qualquer ajuste de parâmetros em `B0`.

---

## 2. Veredito de Congelamento de `B0-rest` (`GO / NO-GO`)

- **Veredito:** **GO para congelamento de `B0-rest` = `A1+A2a+A_scaler`**.
- **Enquadramento científico obrigatório:** **`B0-rest` é um `SQR-HET` adaptado e corrigido**, não uma reprodução integral do SQR original, dadas as diferenças documentadas em [B0_FROZEN_MANIFEST.json](B0_FROZEN_MANIFEST.json):
  1. `virtual_alpha = 0.0` (remoção de `R12`, mantendo `P_penalty=2`, pinball real, bootstrap, SLAs e `R19` intactos);
  2. `mean_field = False` (remoção de `z4/z5`, pois sua reativação em `A4` com scaler corrigido degradou ou não trouxe ganho consistente no controlador independente);
  3. `--empty_slice_guard` (supressão pré-decisão de propostas em slices ociosas);
  4. `--fix_scaler_index` (correção `"emu" -> "eum"` no `FeatureScaler`);
  5. `--urllc_delayed_update` e margens heterogêneas normalizadas sobre `sla_manifest.yaml` `v1.1-g4b` (`C=53` efetivo na telemetria MAC, `M0=10`, `M ∈ [2, 50]`, `gamma=1`).

---

## 3. Contrato de Congelamento e Paridade Causal (6 Propriedades Verificadas)

Todas as 6 propriedades de integração exigidas antes de qualquer `A5` / `D3-v2` foram implementadas em [`freeze_contract.py`](../../../ns-o-ran-gym/src/nsoran/sqr_jc/freeze_contract.py) + [`sqr_run.py`](../../../ns-o-ran-gym/examples/sqr_run.py) e validadas em [`test_sqr_jc_freeze_contract.py`](../../../ns-o-ran-gym/tests/test_sqr_jc_freeze_contract.py) na suíte [`run_sqr_tests.sh`](../../../ns-o-ran-gym/tests/run_sqr_tests.sh):

1. **Paridade bit-a-bit com vs. sem `--causal_trace_path`:** `test_causal_trace_strict_causality_and_bit_exact_parity` confirma `decisions.jsonl`, `step_metrics.csv` e estado final idênticos.
2. **Snapshot pré-ação e estritamente sem dados futuros:** `test_causal_trace_non_anticipation_and_future_divergence` comprova que, sob dois futuros físicos divergentes após a janela $W=25$, todos os snapshots gravados até $W=25$ (`state`, `per_slice_model`, `f_candidates`, `scores_candidates`, `sigma_candidates`, ações) são bit-idênticos, e `verify_causal_snapshot_matches_controllers` valida hashes por slice e `capacity_C`.
3. **Compatibilidade de versões e semântica de features:** `build_version_manifest`, `build_b0_config_descriptor` e `verify_resume_contract` rejeitam qualquer divergência de `mean_field`, `fix_scaler_index`, `virtual_alpha`, `empty_slice_guard`, `margin_bounds`, `urllc_delayed_update`, `config_sha256` ou `payload_sha256` adulterado.
4. **Impossibilidade de reutilizar checkpoint futuro em decisões passadas:** `sqr_run.py` grava `checkpoint_horizon = {max_step_index, max_window_index, max_sim_time_ms}` e `verify_causal_replay_no_future_checkpoint` bloqueia qualquer checkpoint com horizonte $\ge W$ ou sem metadado de horizonte no replay contrafactual.
5. **Identidade do estado físico no replay `A/A`:** `test_physical_state_identity_in_aa_replay_branches` confirma trajetórias físicas (`step_metrics.csv`) bit-idênticas entre ramos `A/A` e prefixo físico bit-idêntico até $t-1$ em ramos `A/B`.
6. **Zero atualizações espúrias em `freeze=True`:** `test_freeze_no_learning_and_resume_source_preservation` + `verify_freeze_no_learning` confirmam imutabilidade total de `support_points`, `alphas`, `tau`, `margin_m` e arquivo de origem sob `--resume --freeze`.

**Distinção metodológica explícita (`D3-v2`/`A5` vs. `--freeze --resume`):**
- *Comparação contrafactual de janela única (`D3-v2` / `A5` em $W$):* opera sobre snapshots instantâneos pré-decisão (`CausalTraceWriter`, com horizonte estritamente $< W$) e ramos físicos `A/A` até $W-1$, medindo exatamente o efeito isolado da troca de ação na janela $W$.
- *Avaliação online em malha fechada (`--resume <ckpt_B0> --freeze`):* executa uma política congelada desde `step = 0` em trajetória independente (*out-of-sample*), na qual desvios de decisão alteram naturalmente filas e evolução futura; avalia-se por estatísticas de episódio entre *seeds*, nunca por pareamento janela-a-janela com a trajetória original.

---

## 4. Delimitação do Artigo e Próximo Gate

- A evidência em `seed=2102` mostra que a restauração de `mean_field` não trouxe melhoria consistente para o controlador independente com os escaladores corrigidos, **sem** concluir que *mean-field* seja inadequado em geral. A contribuição candidata principal permanece **`SQR-JC`** (seleção conjunta ciente de suporte e demanda ativa sobre $\mathcal{A}_{\text{common}}$ vs. `B0-rest` congelado, `equal` e `prop`).
- **Próximos passos (`A5` em commit separado):** implementação modular de `A5` e Gate 1 exploratório em dev, mantendo `D3-v2` confirmatória, `PILOT`, `MAIN`, `A2b`, `A3b` e *seeds* `3101–3103` bloqueados até autorização explícita.
