# Plano Versionado de Correção e Isolamento da Infraestrutura Experimental (`sqr-v2`)

> **ID do Documento:** `PLANO_CORRECAO_INFRAESTRUTURA_V2` · **Versão:** `1.1` (`2026-11-17`)  
> **Status:** `APROVADO COM CONDIÇÕES — GO ETAPAS 1 & 2; ETAPA 3A APÓS RELATÓRIO DA ETAPA 2; ETAPA 3B BLOQUEADA`  
> **Escopo:** Correção rigorosa, versionada e fisicamente isolada da infraestrutura experimental (`BUG-03`, `BUG-04`, `BUG-05`, `BUG-01`, `BUG-06`, higiene `BUG-07/08`, revisão normativa separada de TDD/QoS/PHY e protocolo de ablações isoladas do `BUG-02`), mantendo `B0-rest`, manifestos, checkpoints, *replays* e resultados históricos (`v1`) **100% preservados em disco**.  
> **Proibições explícitas nesta etapa:** NÃO implementar `SQR-CER`, DRL, *bandit* ou qualquer novo seletor acoplado; NÃO executar simulações com *seeds* confirmatórias reservadas (`MAIN {3001..3010}`, `OOS >= 3101`, `G4B_VALIDATION {1201..1209}`); NÃO publicar nem alterar branches remotas (`origin/*`) sem autorização específica; NÃO iniciar Etapa 3A antes da entrega e aprovação do relatório da Etapa 2.

---

## Sumário das 5 Etapas e Critérios de Gate

```
[v1 CONGELADA: B0_FROZEN_MANIFEST.json (17/17 SHA-256)] ──► Inventário pré-commit + TAG local (origin/* intocado)
         │
         ▼
[Etapa 1 — GO] Isolamento físico/Git (`exp/sqr-v2-infra`, `scratch/sqr_v2/`, `src/nsoran/sqr_v2/`, `results/sqr_v2/`) + Governança de Seeds
         │ Gate G-V2-1: inventário pré-commit registrado + 17/17 hashes v1 intactos + SQR_V2_MANIFEST.json + 360 testes v1 verdes
         ▼
[Etapa 2 — GO] IPC Handshake v2 (`run_id` + `window_seq`, `BUG-03` + `BUG-05`) + Análise Não-Destrutiva (`BUG-04`)
         │ Gate G-V2-2 + PARADA OBRIGATÓRIA PARA RELATÓRIO:
         │   - testes das 5 classes de falha IPC sob timeout 50 ms estrito (antiga, duplicada, atrasada, incompleta/malformada, outro run_id);
         │   - aprendizado observa estritamente `prb_executed_mac`; validação causal separada da medição temporal 50 ms;
         │   - paridade A/A exata; invariantes BUG-04 (`executed + unallocated == C`); reanálise somente-leitura da v1; 17/17 hashes v1 intactos.
         ▼  [AGUARDA APROVAÇÃO DO RELATÓRIO DA ETAPA 2]
[Etapa 3 — PARCIAL] Subdividida em:
         ├── 3A (somente preparação e testes isolados pós-Gate 2, sem incorporação automática): `BUG-01` (MCS 0->15), `BUG-06` (outage CQI=0), higiene `BUG-07/08` sob `P0`
         └── 3B (BLOQUEADA): Revisão normativa separada (revisar símbolos DL do slot S em `P1` sem adotá-lo ainda; referência inicial `Q0` 5QI=9; corrigir justificativa errônea de 5QI 79)
         ▼
[Etapa 4] Revalidação do SQR Base & Ablações Isoladas do `BUG-02` (`L0`, `L1`, `L2a NORMA`, `L2b Poda`, `L3`)
         │ Gate G-V2-4: critério composto (SLA c/ denominadores explícitos, calibração quantílica, contraste γ, estabilidade pós-B=2000, telemetria causal, confident > 0%)
         ▼
[Etapa 5] Diagnósticos Separados da Premissa Central do `SQR-CER` (`E5-Idle` & `E5-Transfer` — SEM implementar controlador)
           Gate G-V2-5: interpretado estritamente nas condições avaliadas (sem afirmação definitiva global de limitação informacional)
```

---

## Seção 1 — Etapa 1: Preservação Histórica, Isolamento Versionado (`sqr-v2`) e Governança de Seeds

### 1.1 Por que NÃO modificar `scratch/sqr/` ou `src/nsoran/sqr/` in-place
O manifesto congelado [`MODA/docs/sqr/jc_decision/B0_FROZEN_MANIFEST.json`](../sqr/jc_decision/B0_FROZEN_MANIFEST.json) (linhas `11–28`) fixa os `17` hashes SHA-256 de código e binário do `B0-rest`, incluindo:
- `MODA/ns-3-dev/scratch/sqr/sqr-sim.cc` e `sqr-mac-scheduler.{h,cc}`;
- o próprio executável compilado `MODA/ns-3-dev/build/scratch/sqr/ns3.46-sqr-sim-default`;
- `MODA/ns-o-ran-gym/examples/sqr_run.py`;
- `MODA/ns-o-ran-gym/src/nsoran/sqr/{sqr_controller,quantile_regressor,kernel,coordinator}.py`;
- `MODA/ns-o-ran-gym/src/nsoran/sqr_jc/*` e `tests/test_sqr_jc_freeze_contract.py`.

Qualquer edição ou recompilação direta sobre `scratch/sqr/` invalidaria imediatamente os hashes congelados e quebraria a reprodutibilidade exata de `B0-rest`, dos `127` *replays* do `Gate 1` e dos `390` *replays* de `D2-v2`. Da mesma forma, `contrib/nr` (com os 4 patches defensivos em `MODA/docs/patches_nr/`) e `scratch/rslaq` permanecem **estritamente intocados** (`AGENTS.md` §5.1–5.2).

### 1.2 Arquitetura de Isolamento Duplo (`v1-frozen` × `sqr-v2`)

1. **Inspeção Pré-Commit e Isolamento no Git (`AGENTS.md` §5.8):**
   - **Antes de qualquer commit na `main` ou criação de tag:** conferir `git status --short -uall`, `git branch -vv` e registrar formalmente em [`MODA/docs/sqr/v2_infra/ETAPA1_INVENTARIO_PRE_COMMIT.md`](v2_infra/ETAPA1_INVENTARIO_PRE_COMMIT.md) todos os arquivos modificados e não rastreados envolvidos, preservando integralmente alterações e artefatos preexistentes (inclusive `MODA/docs/superpowers/specs/Auditoria.txt`, `README.md`, relatórios de auditoria e arquivos de `Gate 1`/`Gate Contextual`);
   - Consolidar o estado documental auditado na `main` mantendo `origin/main` e demais branches remotas **estritamente intocadas** (proibido `git push` ou qualquer alteração remota sem autorização específica);
   - Criar tag anotada local imutável na `main`: `v1.0-b0-frozen-audit-2026-11-17`;
   - Abrir branch local dedicada para a nova linha experimental: `exp/sqr-v2-infra`.

2. **Isolamento Físico de Código, Binários, Resultados e Manifestos:**
   - **C++ (`ns-3.46-dev`):** novo cenário e escalonador em `MODA/ns-3-dev/scratch/sqr_v2/`:
     - `sqr-v2-sim.cc` (cenário instrumentado com `IPC Handshake v2` na Etapa 2; apenas após aprovação da Etapa 2, recebe flags isoladas da Etapa 3A para `BUG-01`/`BUG-06`);
     - `rslaq-mac-scheduler.{h,cc}` e `sqr-mac-scheduler.{h,cc}` isolados em `scratch/sqr_v2/` (durante a Etapa 2, mantêm a física idêntica à `v1` para paridade `A/A` e isolamento puro do `IPC v2`; na Etapa 3A, somente após GO pós-Etapa 2, é removida a regra `m_dlMcs == 0 -> 15`);
     - Binário compilado em alvo separado `build/scratch/sqr_v2/ns3.46-sqr-v2-sim-default` via `bash MODA/ns-3-dev/scratch/sqr/ns3_gate.sh build sqr-v2-sim` (mantendo o executável `build/scratch/sqr/ns3.46-sqr-sim-default` e seu SHA-256 congelado em `B0_FROZEN_MANIFEST.json` byte a byte idênticos).
   - **Python (`ns-o-ran-gym`):**
     - Novo pacote `MODA/ns-o-ran-gym/src/nsoran/sqr_v2/` (`__init__.py`, `manifest_v2.py`, `ipc_v2.py`, `env_v2.py`, `controller_v2.py`, `campaign_analysis_v2.py`), reutilizando lógica pura testada por import controlado e nunca sobrescrevendo módulos de `src/nsoran/sqr/` cobertos pelo manifesto `B0`;
     - Novo driver `MODA/ns-o-ran-gym/examples/sqr_v2_run.py` e utilitário não-destrutivo `MODA/ns-o-ran-gym/examples/sqr_v2_reanalyze_v1.py`;
     - Nova suíte `MODA/ns-o-ran-gym/tests/test_sqr_v2_ipc_and_metrics.py` e script `MODA/ns-o-ran-gym/tests/run_sqr_v2_tests.sh`.
   - **Diretório de Resultados e Manifesto Tri-Eixo (`SQR_V2_MANIFEST.json`):**
     - Todos os novos experimentos gravam exclusivamente em `MODA/ns-o-ran-gym/results/sqr_v2/<experimento>/`;
     - Todo experimento `v2` gera automaticamente `SQR_V2_MANIFEST.json` com hash SHA-256 dos artefatos de código, separação explícita causal vs. temporal (`ipc_mode`, `ipc_timeout_ms`, `rt_50ms_deadline_enforced`) e separação em **3 eixos ortogonais** (proibindo misturar correção de bug com mudança normativa ou ablação de aprendizado no mesmo passo sem tag distinta):
       - **`infra_rev`:** código de infraestrutura (`ipc_v2`, `metrics_v2`, e somente na 3A+ `mcs_v2`, `snr_v2`, `c_sync_v2`);
       - **`phy_qos_profile`:** configuração física/normativa (`P0_legacy_tdd7_uma300`, `P0_base_uma2000`, etc.; `P1` bloqueado na 3B);
       - **`learning_ablation`:** configuração algorítmica do SQR (`L0_paper_r12_on`, `L1_va0_only`, `L2a_norma_decay`, `L2b_pruning`, `L3_aniso_ls`).

### 1.3 Governança Estrita de *Seeds* (Proteção Contra Vazamento e Overfitting)

| Classe de *Seeds* | Conjunto | Uso Autorizado nas Etapas 1–5 | Status |
|---|---|---|---|
| **Smoke / Fault Injection** | `{101, 102, 103}` (e `21`) | Testes unitários/integração IPC (`BUG-03`), injeção de atrasos `>50 ms`, testes `MCS=0`/`CQI=1`, smoke TDD `P0/P1` | ✅ **Liberada** |
| **Diagnóstico Homogêneo (`2 km` / `300 m`)** | `{1..5}` (subset de `1..10`) | Diagnóstico pareado de `BUG-01`, `BUG-02` (`L0–L3`), `BUG-03` e *sweep* de raio persistido | ✅ **Liberada** |
| **Desenvolvimento Heterogêneo (`300 m`)** | `{2101, 2102, 2103}` | Reavaliação pareada do impacto de `BUG-03` no `B0-rest`, contraste `2102 × 2103` livre de lag, ablações `L0–L3`, diagnóstico `Etapa 5` | ✅ **Liberada** |
| **Calibração Exploratória (`G4-B fit`)** | `{1001..1003, 1101..1106}` | Somente leitura / comparação retrospectiva opcional | ⚠️ **Somente diagnóstico se necessário** |
| **RESERVADAS / CONFIRMATÓRIAS** | **`MAIN {3001..3010}`**, **`OOS >= 3101`**, **`G4B_VALIDATION {1201..1209}`** | **PROIBIDO** usar em qualquer teste, ajuste, ablação ou pré-seleção nas Etapas 1–5 | 🚫 **BLOQUEADA** |

### 1.4 Critério de Conclusão da Etapa 1 (`Gate G-V2-1`)
- `MODA/docs/sqr/jc_decision/B0_FROZEN_MANIFEST.json` verifica **`17/17` hashes SHA-256 exatos (`0` divergências)**;
- `bash MODA/ns-o-ran-gym/tests/run_sqr_tests.sh -x -q` segue passando integralmente (**`360 passed`**);
- Esqueleto `scratch/sqr_v2/` + `src/nsoran/sqr_v2/` + `SQR_V2_MANIFEST.json` compila e registra os 3 eixos sem alterar nenhum arquivo congelado da `v1`.

---

## Seção 2 — Etapa 2 (Prioridade Imediata): Protocolo `IPC Handshake v2` (`BUG-03` + `BUG-05`) e Reprocessamento Analítico (`BUG-04`)

### 2.1 Diagnóstico Preciso das Falhas da `v1` (`BUG-03` e `BUG-05`)
A auditoria no código confirmou quatro falhas acopladas no IPC `v1`:
1. **`sqr-sim.cc:1540-1611`:** `sem_post(g_semMetricsReady)` dispara na linha `1542`, **antes** de `sqr_prb_windows.csv` ser atualizado e receber `flush()` (linhas `1562-1611`), criando janela de corrida na telemetria de capacidade `total_prb`.
2. **`sqr-sim.cc:1685-1723` e `action_controller.py:34-41`:** o Python anexa indefinidamente em `rslaq_actions_for_ns3.csv`, e o ns-3 relê o arquivo inteiro do byte `0` a cada `100 ms` ($O(T^2)$), sobrescrevendo `dedicatedPrb[sliceId]` em memória com o último registro sem validar `timestamp` ou `window_seq`.
3. **`sqr_controller.py:300-301`:** `score_candidates` chama `predict(X)` e `uncertainty(X)` em duas passadas separadas (dobro do custo da matriz Matérn `(54 × 2000 × 7)`), levando `decision_time_ms` até `97,3 ms` (`> 50 ms`).
4. **`sqr-sim.cc:1670-1803` + `ns_env.py:445-469`:** quando `sem_timedwait(50 ms)` estoura no C++, o timeout só emite `NS_LOG_WARN`, mantém a cota anterior e não drena o `controlSemaphore`. O `sem_post` atrasado do Python permanece no semáforo; no passo seguinte ($w+1$), o C++ consome instantaneamente a ação velha do passo $w$ e **dessincroniza permanentemente toda a corrida em 1 janela**, enquanto o Python continua treinando com `prb_executed` (pós-$\Pi$) que o MAC nunca aplicou! Ademais, em [`sqr_run.py:615, 1006-1021`](../../ns-o-ran-gym/examples/sqr_run.py#L615), o Passo 1 sincroniza `controllers.n_prbs` de `52 -> 53` **após** `act()` (`BUG-05`).

### 2.2 Especificação do Contrato `IPC Handshake v2` (`sqr-v2-sim.cc` ↔ `sqr_v2/ipc_v2.py` + `env_v2.py`)

#### A. Identificação Inequívoca de Execução (`run_id`) e Janela (`window_seq`) — Ordem Atômica no C++
Antes de sinalizar `semMetricsReady` na fronteira $w \ge 1$ (`sim_time_ms = w * period_ms`):
1. **Drenar e persistir telemetria MAC** da janela encerrada $(w-1, w]$ em `sqr_prb_windows.csv` (mantendo colunas base e adicionando `run_id,window_seq,applied_action_seq,ipc_status,decision_wall_ms,over_50ms`);
2. **Escrever `sqr-kpms.txt` atomicamente** (grava em `sqr-kpms.txt.tmp`, `flush()`, fecha e executa `std::filesystem::rename`);
3. **Escrever `sqr_ipc_indication_v2.json` atomicamente** (`.tmp` $\to$ `rename`) identificando inequivocamente execução e janela:
   ```json
   {
     "protocol_version": 2,
     "run_id": "sqrv2-seed101-...",
     "window_seq": 1,
     "sim_time_ms": 100,
     "period_ms": 100,
     "total_rbg": 53,
     "total_prb": 53,
     "active_ues": 0,
     "kpm_file": "sqr-kpms.txt",
     "governed_prev_window_seq": 0,
     "prev_applied_action_seq": 0,
     "prev_ipc_status": "INIT",
     "prev_prb_executed_rbg": [18, 18, 17],
     "prev_decision_wall_ms": 0.0,
     "recent_windows_tail": []
   }
   ```
   *Efeito direto contra `BUG-05`:* mesmo com `active_ues == 0` no `env.reset()` ($w=1$), `total_prb = 53` e `run_id` já vêm no envelope antes da 1ª decisão (`sqr_v2/env_v2.py` lê o envelope no `reset()` e sincroniza `total_prbs` antes de `act()`).
4. **Drenar tokens e arquivos obsoletos ANTES de postar a nova indicação:**
   - Consumir quaisquer tokens pendentes em `g_semControl` (`while (sem_trywait(g_semControl) == 0) g_ipcLateTokensDrained++;`) e remover/classificar qualquer `sqr_ipc_action_v2.csv` remanescente em disco antes de postar `g_semMetricsReady`.
   Isto garante que um `sem_post` atrasado ocorrido *entre* o timeout anterior e a nova fronteira jamais seja consumido como se fosse a resposta da janela $w$.
5. **Só então postar** `sem_post(g_semMetricsReady)`.

#### B. Resposta Identificada $O(1)$ (`run_id` + `window_seq` + `# END_V2`) e Redução de Custo no Python
1. **`KernelQuantileRegressorV2.predict_and_uncertainty(X)`:** calcula a matriz kernel `K = self.kernel(X, self._sx)` **uma única vez**, retornando simultaneamente `(K @ alpha, 1.0 - K.max(axis=1))` (~`50%` de redução no tempo de inferência de `score_candidates`, sem perda numérica).
2. **Escrita atômica de tamanho constante ($O(1)$) em `sqr_ipc_action_v2.csv`:**
   - Para a indicação `(run_id, window_seq == w)`, o Python grava em `sqr_ipc_action_v2.csv.tmp`, executa `flush()` + `os.replace(..., "sqr_ipc_action_v2.csv")` **somente com o cabeçalho, as $L=3$ linhas da janela corrente e o marcador de integridade `# END_V2`**:
     `run_id,window_seq,sim_time_ms,slice_id,prb_proposed,prb_requested,prb_allocated,dedicated_pct,algorithm,agent_decision_wall_ms`
   - Um log histórico separado (`SqrActions_v2_history.csv`, append-only) guarda todas as decisões para auditoria, **nunca lido pelo ns-3**.

#### C. Loop de Espera com Validação Estrita (5 Classes de Falha) e Timeout Explícito no C++
Na mesma fronteira $w$ (após `sem_post(g_semMetricsReady)`):
1. **Separação estrita entre conformidade temporal de `50 ms` e validação causal:**
   - `--ipcTimeoutMs` tem **default `50` ms** em tempo de parede;
   - O modo `--ipc-mode strict_rt` (`ipcTimeoutMs = 50`) mede conformidade real ao orçamento de `50 ms` e conduz os testes de injeção de atrasos/falhas;
   - O modo opcional `--ipc-mode causal_sim` (`ipcTimeoutMs = 2000`) existe **exclusivamente** para simulações causais livres de jitter de CPU do host (`rt_50ms_deadline_enforced = false`), sendo **proibido** utilizá-lo como evidência de atendimento ao prazo de `50 ms`. Ambos os modos registram `agent_decision_wall_ms` (`p50/p95/p99/max`) e taxa `> 50 ms` independentemente.
2. Executar laço `while (true)` até `deadline_abs = now + ipcTimeoutMs`:
   - Chamar `sem_timedwait(g_semControl, &deadline_abs)` (tratando `EINTR` sem sair antes do prazo);
   - Se `errno == ETIMEDOUT`:
     - **Ação explícita de timeout (`TIMEOUT_HOLD`):** manter a cota vigente no MAC, incrementar `g_ipcTimeoutHoldCount`, gravar `prev_ipc_status = "TIMEOUT_HOLD"` (sem alterar `prev_applied_action_seq`) e encerrar a espera desta janela;
   - Se `ret == 0`: consumir atomicamente `sqr_ipc_action_v2.csv` do disco para memória e validar nesta ordem:
     - **(i) Incompleta / Truncada / Malformada:** arquivo ausente (post espúrio), ausência do marcador `# END_V2`, colunas faltantes, `NaN`/Inf ou ausência de todas as $3$ slices `{0,1,2}` $\implies$ incrementar `g_ipcMalformedDiscardedCount`, rejeitar e continuar aguardando até `deadline_abs`;
     - **(ii) De outra execução (`run_id != g_runId`):** incrementar `g_ipcWrongRunDiscardedCount`, rejeitar e continuar aguardando até `deadline_abs`;
     - **(iii) Duplicada (`action.window_seq == g_lastAppliedActionSeq`):** incrementar `g_ipcDuplicateDiscardedCount`, rejeitar e continuar aguardando até `deadline_abs`;
     - **(iv) Antiga / Obsoleta (`action.window_seq < w`):** incrementar `g_ipcObsoleteDiscardedCount`, emitir warning com `action.window_seq` vs `w`, rejeitar e continuar aguardando até `deadline_abs`;
     - **(v) Futura ou divergência de timestamp (`action.window_seq > w || action.sim_time_ms != nowMs`):** incrementar `g_ipcFutureOrMismatchCount`, rejeitar e continuar aguardando até `deadline_abs`;
     - **Válida (`run_id == g_runId && window_seq == w && sim_time_ms == nowMs`):** aplicar `ApplyQuotaAction` no MAC, atualizar `prev_applied_action_seq = w`, `prev_ipc_status = "APPLIED"`, `prev_prb_executed_rbg`, drenar eventual duplicata já presente no semáforo (`sem_trywait`), e sair do `while`.
3. Ao fim da simulação, emitir no stdout e em `<outputDir>/sqr_ipc_v2_summary.json` todos os contadores e percentis `decision_wall_ms` (`applied`, `timeout_hold`, `obsolete_discarded`, `duplicate_discarded`, `malformed_discarded`, `wrong_run_discarded`, `late_tokens_drained`, `over_50ms_count`, `p50/p95/p99/max`).

#### D. Reconciliação Causal Obrigatória no Aprendizado Python (`sqr_v2_run.py`)
Quando o Python retorna de `_wait_data_availability()` na fronteira $w+1$:
1. Drena eventuais posts coalescidos do semáforo até alinhar com `sqr_ipc_indication_v2.json` corrente e valida `indication["run_id"] == expected_run_id` e `governed_prev_window_seq == w`;
2. **Treinamento causal fiel ao que o MAC realmente executou:**
   - Lê `prb_executed_mac = receipt["prev_prb_executed_rbg"]` (cota confirmada pelo MAC para a janela $w$);
   - Se `prev_ipc_status == "APPLIED"` (`prev_applied_action_seq == w`): o aprendiz recebe `observe(z_w, prb_executed_mac, y_{w+1})` e (se habilitada na ablação `L0`) atualizações virtuais sobre cortes de $\Pi$;
   - Se `prev_ipc_status == "TIMEOUT_HOLD"`: o agente registra no `step_metrics.csv` ambos `prb_requested_ipc` (proposta enviada com atraso/perdida) e `prb_executed_mac` (`prev_prb_executed_rbg` mantido pelo MAC durante a janela $w$), **suprime `virtual_updates` sobre a proposta não aplicada** e garante que o aprendizado `observe(z_w, a_{\text{MAC}}, y_{w+1})` **nunca** utilize a ação rejeitada pelo timeout (`prb_executed_mac` ou skip explícito com `--learn-on-timeout-hold=false`).
3. Nota de escopo para o caminho E2: o port `v2` do `xapp_sqr_v2.py` (incluindo `learned`/`virtual_updates` em `DEFERRED_COLS`, modos das 3 slices e `run_id`/`window_seq`) fica preparado para integração posterior à conclusão dos gates do caminho IPC.

### 2.3 Correção Não-Destrutiva de `BUG-04` (`campaign_analysis_v2.py` + `sqr_v2_reanalyze_v1.py`)
Sem tocar em nenhum CSV bruto em `results/sqr_campaign/` ou `results/sqr_jc_decision/`:
1. Em `campaign_analysis_v2.py`, corrigir a agregação celular de `prb_idle_est` de `.groupby("step")[col].sum(min_count=1)` para `.groupby("step")[col].first()` (com asserção de teste `rb_cell_total_executed_mean + rb_cell_total_idle_est_mean == total_prbs` por janela).
2. Separar em dois painéis e dois conjuntos de colunas/tabelas independentes (eliminando a mistura de unidades da Fig. 4):
   - **Taxa por slot (`PRBs / slot`, escala `[0, C]`):** `prb_requested`, `prb_executed`, `prb_unallocated_quota = C - Σ prb_executed` (ociosidade de cota decidida pela política), `prb_scheduled_per_active_slot = Σ prb_scheduled_rbg / active_dl_slots` e `prb_scheduled_per_nominal_slot = Σ prb_scheduled_rbg / nominal_dl_slots`;
   - **Volume acumulado na janela (`RBG·slots / janela de 100 ms`):** `prb_scheduled_window_total`, `prb_budget_window_total` e `prb_idle_window_total` lidos diretamente de `sqr_prb_windows.csv`.
3. Disponibilizar `examples/sqr_v2_reanalyze_v1.py` para gerar **errata sombreada somente-leitura** em `results/sqr_v2/reanalysis_v1_bug04/` a partir dos CSVs brutos da campanha `v1` de `2 km`, deixando os arquivos originais em `results/sqr_campaign/` intocados.

---

## Seção 3 — Etapa 3: Fidelidade do Cenário Físico (Separação Estrita `3A` vs `3B`)

> **Princípio de isolamento causal e governança de autorização:**  
> - **Etapa 3A (`AGUARDANDO RELATÓRIO E GO PÓS-ETAPA 2`):** autorizada **apenas** para preparação e testes isolados depois da entrega e aprovação do relatório da Etapa 2, **sem** incorporação automática ao baseline de referência antes da aprovação dos gates anteriores.  
> - **Etapa 3B (`BLOQUEADA`):** permanece integralmente bloqueada até reanálise normativa específica dos símbolos DL no slot especial `S` e do mapeamento QoS/5QI.

### 3.1 Etapa 3A — Preparação e Testes Isolados de `BUG-01` (MCS), `BUG-06` (SNR/CQI Outage) e Higiene (`BUG-07/08`) [Somente após Gate 2]

1. **`BUG-01` — Remoção do override `m_dlMcs == 0 -> 15` em `SqrV2MacScheduler`:**
   - **Localização na `v1`:** `MODA/ns-3-dev/scratch/sqr/rslaq-mac-scheduler.cc:1553-1556`, chamado por `sqr-mac-scheduler.cc:504`.
   - **Alteração na `v2` (`scratch/sqr_v2/sqr-v2-mac-scheduler.{h,cc}`):**
     - Sobrescrever `BeforeDlSched` para espelhar `NrMacSchedulerOfdmaPF::BeforeDlSched` (`contrib/nr/model/nr-mac-scheduler-ofdma-pf.cc:147-153`): atualizar `CalculatePotentialTPutDl(assignableInIteration)` e `m_lastDlBufferSize[rnti]`, **removendo completamente** `if (ueInfo->m_dlMcs == 0) { ueInfo->m_dlMcs = 15; }`;
     - Expor atributo booleano `LegacyForceMcs15OnZero` (default `false`; controlável via `--legacyForceMcs15OnZero=false|true`) exclusivamente para permitir teste pareado `A/B` que isole o impacto exato de `BUG-01`;
     - **Preservação comprovada dos valores legítimos:**
       - `CQI = 1` (`wbCqi = 1`, `SNR ≈ -2,18 dB`) $\implies$ `NrAmc::GetMcsFromCqi(1) = 0` (`QPSK`, `SE = 0,15`) **permanece `MCS = 0`**;
       - Pré-primeiro CQI ou expiração (`RefreshDlCqiMaps`: `m_wbCqi = 1`, `m_dlMcs = StartMcsDl = 0`) mantém `MCS = 0` conservador até o próximo relatório CQI;
       - `CQI ∈ {2..15}` mantém exatamente `MCS ∈ {2..28}` sem alteração.
2. **`BUG-06` — Tratamento honesto de `CQI = 0` (outage) vs ausência transitória de CQI pré-attach:**
   - Em `sqr-v2-sim.cc`, distinguir quando o UE possui RNTI e `wbCqi == 0` pós-attach (`cqiWideband = 0`, `snrDb = -10.0` ou flag explícita de outage) de quando não há relatório pré-attach (`cqiWideband = -1`, `snrDb = -999.0`);
   - Em `sqr_v2/kpis_v2.py`, com `--snrOutagePolicy=zero_floor` (default `v2`; vs `legacy_impute` opcional para ablação), UEs em outage `CQI = 0` recebem **`snr_norm = 0.0` (`-10,0 dB`)** e **nunca** são mascarados com `15 dB` ou com a mediana dos UEs próximos à antena.
3. **Higiene complementar (`BUG-07`, `BUG-08`, `C=53`):**
   - Exigir `ue.attached` no loop de emissão KPM (`sqr-v2-sim.cc`) e encerrar explicitamente `clientApp` quando `UeDeparture` disparar em sessões `< 0,6 s` (`BUG-07`);
   - Registrar em `sqr_prb_windows.csv` tanto `active_dl_slots` (slots com `AssignDLRBG` chamado) quanto `nominal_dl_slots` (slots DL/S do padrão TDD na janela, `BUG-08`);
   - Nota de fidelidade documentada sobre `BUG-08` (ordenação PF intra-slot) e `BUG-09` (`UpdatePeriod = 0`): manter `SortUeVectorByAlgorithm` 1×/slot e `UpdatePeriod = 0` como **padrão da 3A** (ou expor flags explícitas `--perRbgPfSort=false` e `--channelUpdatePeriodSec=0`) para não introduzir duas mudanças de dinâmica ao mesmo tempo que `BUG-01`.

### 3.2 Etapa 3B (`BLOQUEADA`) — Revisão Normativa: TDD, Contabilização do Slot Especial `S` e QoS `5QI`

Esta etapa **permanece BLOQUEADA** e **não** deve ser codificada nem incorporada automaticamente. Antes de qualquer desbloqueio futuro, ficam registradas as seguintes correções técnicas obrigatórias sobre a proposta inicial da `v1.0`:

#### A. Revisão da Contabilização de Símbolos DL no Slot Especial `S` (`P1` NÃO adotado definitivamente ainda)
Na `v1`, o padrão `"D|D|8D|4GB|4U|U|U"` vira `"DL|DL|DL|S|UL|UL|UL|"` (`7 ms`), que **viola** a enumeração `dl-UL-TransmissionPeriodicity` da TS 38.331 (`{0.5, 0.625, 1, 1.25, 2, 2.5, 5, 10}` ms + `{3, 4}` ms v1530), aborta `AssignCsiRsOffset` (`nr-gnb-phy.cc:1830-1833`) e gera batimento `100 mod 7 = 2`.

Contudo, a tabela da `v1.0` tratava o slot `S` como equivalente a `1,0` slot DL cheio (`60,0% DL` em `P1 = DL|DL|S|UL|UL|`). Essa contabilização é **imprecisa** sob dois referenciais que precisam ser explicitamente distinguidos antes de escolher um perfil:
1. **Na implementação atual do 5G-LENA (`nr-mac-scheduler-ns3.cc:1962-1984`):** todo slot `DL` usa `1` símbolo de DL Ctrl e deixa `13` símbolos PDSCH; o slot `S` adiciona `1` símbolo DL Ctrl no início e `1` símbolo UL Ctrl no fim (`GP = 0` no scheduler TDMA idealizado), restando **`12` símbolos PDSCH** (`92,3%` de um slot DL). Assim, em `100 ms` (`1400` símbolos OFDM a `μ=0`):
   - **`P0` (`DL|DL|DL|S|UL|UL|UL|`, `7 ms`):** média de `14,2857 × (3×13 + 12) = 728,57` símbolos PDSCH (`52,04%` dos `1400` símbolos OFDM);
   - **`P1` (`DL|DL|S|UL|UL|`, `5 ms`):** `20 × (2×13 + 12) = 760` símbolos PDSCH (`54,29%` dos `1400` símbolos OFDM — acréscimo real de `+4,3%` relativo em símbolos PDSCH úteis na física do 5G-LENA, e não `+5%` sobre base de `60%` cheia).
2. **Na especificação normativa 3GPP (`TS 38.331 §6.3.2` `TDD-UL-DL-Pattern` + `TS 38.213 §11.1`):** um slot especial `S` contém `nrofDownlinkSymbols` + período de guarda `GP >= 2` + `nrofUplinkSymbols` (por exemplo, formato `10:2:2` $\implies$ apenas `9` símbolos PDSCH após `1` DL Ctrl, totalizando `700/1400 = 50,0%` em `P1`; ou formato `6:4:4` $\implies$ `5` símbolos PDSCH, `620/1400 = 44,3%` em `P1`).
- **Decisão atual (`BLOQUEADA`):** `P1` **não** é adotado definitivamente ainda; a análise comparativa completa em nível de símbolo (`P0`, `P1` com diferentes razões de `S` no 5G-LENA vs 3GPP, e `P2` `DDDSU`) será submetida ao usuário apenas após a conclusão das Etapas 2 e 3A.

#### B. Fronteira Explícita de QoS 5G (`TS 23.501`): Referência Inicial `Q0` e Retratação sobre `5QI 79`

| Aspecto | Estado Normativo Revisado (`3B BLOQUEADA`) | Correção Técnica Obrigatória |
|---|---|---|
| **Mapeamento `5QI` inicial (`Q0`)** | **Manter `Q0` (`5QI = 9`, `NGBR_VIDEO_TCP_DEFAULT`, portador único padrão, com abstração declarada) como referência inicial.** | A justificativa anterior da `v1.0` sugerindo `5QI = 79` para mMTC estava **incorreta**: conforme **3GPP TS 23.501 Table 5.7.4-1** e [`nr-qos-flow.h:100`](../../ns-3-dev/contrib/nr/model/nr-qos-flow.h#L100) / [`nr-qos-flow.cc:126`](../../ns-3-dev/contrib/nr/model/nr-qos-flow.cc#L126), **`5QI = 79` é `NGBR_V2X` (mensagens V2X não-GBR, prioridade 65, PDB 5 ms no modelo Rel-19 do 5G-LENA / 50 ms na TS 23.501)** — não representando tráfego mMTC/IoT massivo de longo atraso. Qualquer futura opção `Q1` exigirá seleção fundamentada na TS 23.501 e validação isolada após desbloqueio da 3B. |
| **Habilitadores PHY/MAC URLLC** (mini-slots Tipo B, `DCI 2_1` preempção, $\mu \ge 1$, duplicação PDCP) | **Mantidos fora (abstração explícita).** | Avalia controle inter-slice no Near-RT RIC em `100 ms` sobre portadora $\mu=0$ (`C=53`). |
| **Semântica MAC de Cota** (`Dedicated = Min = Max`, DL-only, não *work-conserving*) | **Mantida idêntica ao `R27`.** | Sem isolamento rígido sem redistribuição automática no MAC, a economia espectral ($\gamma \cdot a/C$) e a transferência explícita tornam-se inobserváveis sob tráfego *greedy*. |
| **Janela de Avaliação SLA** (`100 ms` eMBB/URLLC + `1 s` rolling mMTC vs `2000 ms` `TS 23.501`) | **Mantida (`100 ms` Near-RT + `1 s` mMTC), com declaração explícita no texto.** | Reflete o horizonte de controle do Near-RT RIC (`10–1000 ms`), distinto da janela média GFBR de `2000 ms` da TS 23.501. |
| **Parâmetros Rádio e Topologia** (`R=300 m`, `UMa`, `30 dBm`, `1×1`, `10 MHz / C=53`) | **Em avaliação futura da 3B (não alterar ainda na 2/3A).** | Separa falha de cobertura RF de contenção inter-slice. |

---

## Seção 4 — Etapa 4: Revalidação do SQR Base e Ablações Isoladas de `BUG-02` (Sem `SQR-CER`)

### 4.1 Por que NÃO aplicar tudo junto para `BUG-02`
Na `v1`, `BUG-02` decorreu da interação de três decisões de projeto (`alpha_virt = -0.05` [`R12`], kernel largo `l = 1.0` [`R08`] e poda `|α_j|·k` sem decaimento [`R10`]), agravadas por `BUG-01` e `BUG-03`. Além disso, em `B0-rest` (`300 m`), a simples desativação de `R12` (`virtual_alpha = 0.0`) já removeu o colapso catastrófico de $f_\tau$, permitindo decisões `confident`. Alterar simultaneamente regularização NORMA e *length-scales* esconderia qual mecanismo realmente resolve o viés na `v2`.

### 4.2 Matriz Sequencial de Ablações Isoladas (`infra_rev = ipc_v2 + mcs_v2 + snr_v2 + metrics_v2`)

| ID Ablação | Mudança Única vs Anterior | Hipótese Testada | Seeds de Dev |
|---|---|---|---|
| **`L0` (`SQR-Literal-v2`)** | `virtual_alpha = 0.05` (`R12` ligado), `length_scale = 1.0`, poda `I_j` sem NORMA | Verifica se, com canal são (`BUG-01`/`06` corrigidos) e `IPC v2` (`BUG-03`), `R12 = 0.05` ainda colapsa $f_\tau < 0$ em `300 m` e `2 km` | `{1, 2, 3}` (`2 km` & `300 m` hom.) + `{2101, 2102, 2103}` (`300 m` het.) |
| **`L1` (`R12-Off`, Base de Referência Recomendada)** | Apenas **`virtual_alpha = 0.0`** (mantém `l = 1.0`, poda `I_j`, sem NORMA) | Isola o ganho limpo da desativação de `R12` sem lag IPC (especialmente reavaliando a queda `-27 p.p.` da `seed 2103` livre dos `42–47%` de desync IPC!) | `{1, 2, 3}` (`300 m` hom.) + `{2101, 2102, 2103}` (`300 m` het. $\lambda \in \{2,5,6\}$) |
| **`L2a` (`NORMA-Decay`)** | Introduz **apenas** encolhimento de pesos NORMA $(1 - \eta\lambda_{\text{reg}})\alpha_j$ (grade $\lambda_{\text{reg}} \in \{10^{-3}, 5\times 10^{-3}\}$) mantendo a **mesma política de poda `I_j`** | Isola o efeito exclusivo da regularização NORMA na deriva negativa após encher o buffer $B = 2000$ (`t > 200 s`) | Mesmas seeds pareadas da `L1` |
| **`L2b` (`Pruning-Policy`)** | Altera **apenas** a política de poda na memória $B = 2000$ (FIFO / balanço por sinal ou idade), **sem** alterar $\lambda_{\text{reg}}$ | Separa o efeito da regra de substituição/poda de vetores de suporte do efeito da regularização NORMA | Mesmas seeds pareadas da `L1` |
| **`L3` (`Aniso-Action-Scale`)** | Ajusta apenas a escala da coordenada de ação $a/C$ para $\ell_a \in \{0{,}20;\,0{,}35\}$ (mantendo $\ell_z = 1{,}0$) sobre a melhor entre `L1`/`L2a`/`L2b` | Testa se localizar o kernel na ação evita vazamento entre ações altas e baixas, melhorando $\partial f_\tau / \partial a$ e o contraste $\gamma=0$ vs $\gamma=1$ | Mesmas seeds pareadas da `L1` |

### 4.3 Critérios Compostos de Conclusão da Etapa 4 (`Gate G-V2-4` — Pré-requisito da Etapa 5)
A proporção de decisões `confident` **não** pode ser usada como critério único de sucesso (`confident > 0%` é apenas condição de observabilidade, nunca suficiente isoladamente). O avanço exige avaliação conjunta dos cinco indicadores abaixo:
1. **Desempenho SLA com denominadores explícitos (`R19`):** reportar obrigatoriamente `STRICT_ALL3_R19` (`cond_pass_pct`, `uncond_pass_pct`) e taxas por slice acompanhadas das contagens exatas `PASS`, `FAIL` e `UNKNOWN` (sem ocultar cenários por cobertura parcial);
2. **Integridade causal e zero desync:** `ipc_timeout_hold == 0`, `ipc_obsolete_discarded == 0`, e `100%` de concordância `prb_requested_ipc == prb_executed_mac` em todas as execuções da campanha limpa (além da paridade causal documentada sob injeção de timeouts);
3. **Diferenciação real de $\gamma \in \{0, 1\}$ e eficiência de PRBs:** em execuções pareadas nas mesmas *seeds* com ativação `confident > 0%`, a variante com custo ($\gamma = 1$) consome menos PRBs médios ($\overline{a}_{\gamma=1} < \overline{a}_{\gamma=0}$) com balanço explícito contra eventual custo em violações SLA;
4. **Calibração empírica associada aos desfechos corretos:** o erro $\mathbb{P}(y_{w+1} < f_\tau(z_w, a_w^{\text{exec}}))$ calculado estritamente sobre `governed_prev_window_seq == w` satisfaz o viés conservador da Seção IV-D do artigo base;
5. **Estabilidade pós-saturação da memória ($B = 2000$):** ausência de colapso quantílico ou degradação de cobertura entre `t ≤ 200 s` (memória crescente) e `t > 200 s` (memória saturada em $2000$ vetores).

---

## Seção 5 — Etapa 5: Teste da Premissa Central do Futuro `SQR-CER` (Apenas Diagnóstico — Sem Implementação)

### 5.1 Pergunta Científica Única da Etapa 5
Antes de projetar ou codificar qualquer controlador `SQR-CER`, responder empiricamente no ambiente `v2` revalidado (Etapa 4 aprovada):
> **As predições quantílicas $\hat{f}_\tau^{(i)}(z_i, a)$ e incertezas $\sigma_i(z_i, a)$ da variante de referência SQR `v2` ordenam corretamente os efeitos reais observados ao alterar a alocação de PRBs — tanto ao aproveitar capacidade ociosa quanto ao transferir PRBs entre slices sob célula cheia?**

### 5.2 Protocolo em Dois Eixos Separados (Sem Aprender em Replays Futuros)

1. **`E5-Idle` — Teste de Aproveitamento de Capacidade Realmente Ociosa:**
   - Selecionar estados estratificados (nas seeds de dev `{2101, 2102, 2103}`, excl. confirmatórias) onde $\sum_i a_i^{\text{base}} < C$ ou uma slice recebe cota $> 0$ sem fila/UE ativo (`prb_unused_quota > 0` ou `prb_unallocated_quota > 0`);
   - Avaliar via *replays* contrafactuais `v2` de janela única se atribuir os PRBs ociosos à slice com maior déficit/menor margem prevista $\hat{m}_i(z_i, a_i)$ ou maior ganho marginal previsto $\Delta \hat{f}_\tau^{(i)} = \hat{f}_\tau^{(i)}(z_i, a_i + \Delta_{\text{idle}}) - \hat{f}_\tau^{(i)}(z_i, a_i)$:
     - Aumenta `STRICT_ALL3_R19` frente ao baseline sem introduzir dano (`Harm <= 1` ocorrência / taxa próxima de zero);
     - Ordena corretamente quais janelas respondem a PRBs extras vs quais são insensíveis à cota (cauda não-monotônica/TDD).

2. **`E5-Transfer` — Teste de Transferência Inter-Slice sob Contenção ($\sum_i a_i = C$):**
   - Para estados onde há violação de SLA em pelo menos uma slice receptora $i$ sob célula saturada, enumerar uma grade pré-registrada de transferências $(j \xrightarrow{\delta} i)$ com $\delta \in \{2, 4, 8\}$ PRBs a partir de slices doadoras $j \ne i$;
   - Executar *replays* contrafactuais `v2` estritamente causais (`checkpoint` com `max_window_index < w` — **proibido** consultar desfechos futuros durante a decisão!) e medir **duas propriedades ordinais objetivas**:
     - **Segurança da slice doadora $j$:** o critério $\hat{f}_\tau^{(j)}(z_j, a_j - \delta) \ge \Lambda_j$ (e sua versão LCB com $\sigma_j$) separa transferências que **preservam** o SLA do doador daquelas que o quebram? (Métricas: `AUROC_donor`, *False Safe Donor Rate* $\le 10\%$, *Recall* de doadores seguros);
     - **Poder de ordenação de ganho na receptora $i$:** entre doações seguras, o ganho previsto $\Delta \hat{f}_\tau^{(i)}(+\delta)$ correlaciona positivamente na ordem de postos (Spearman $\rho$, *top-1 / top-2 regret* contra a melhor intervenção da grade no *replay*) com a melhora real observada $\Delta y_{w+1}^{(i)}$?

3. **Regra sobre *replays* antigos vs. novos:**
   - Os `127` *replays* de `Gate 1` e `390` de `D2-v2` (linha `v1`) servem **apenas como diagnóstico exploratório offline**;
   - A confirmação da premissa na Etapa 5 exige obrigatoriamente **novos *replays* executados no binário `sqr-v2-sim` limpo**, verificados com teste `A/A` bit-exato antes da adjudicação e **sem nenhum mecanismo de aprendizado lendo resultados futuros**.

### 5.3 Critério de Decisão Pós-Etapa 5 (`Gate G-V2-5`) e Escopo de Interpretação
- **Preservação da separação `E5-Idle` vs `E5-Transfer`:** os dois diagnósticos são mantidos e adjudicados separadamente (ganho em aproveitamento de ociosidade não compensa nem substitui falta de poder discriminativo em transferências inter-slice sob contenção).
- **Critério de avanço para futura especificação do `SQR-CER`:** (i) em `E5-Transfer`, correlação ordinal positiva estatisticamente consistente (Spearman $\rho > 0$ nos estados responsivos) e *False Safe Donor Rate* controlada ($\le 10\%\text{–}15\%$) com `confident`; e (ii) em `E5-Idle`, ganho líquido observável sobre o baseline estático **sem** introduzir dano relevante nem desperdiçar recursos quando `n_ues == 0`.
- **Interpretação estritamente circunscrita em caso de resultado negativo:** caso a premissa não demonstre poder ordinal suficiente mesmo após as correções da `v2`, **o resultado negativo não autoriza afirmar uma limitação informacional definitiva em abstrato** — ele deve ser interpretado **exclusivamente dentro das condições efetivamente avaliadas** (representação de estado $z_t$ testada, cadência de `100 ms`, estimador quantílico SQR/hiperparâmetros avaliados e regime físico/tráfego específico), motivando investigar explicitamente se o limite decorre da escolha de features, da formulação do estimador ou das variáveis disponíveis na janela `100 ms`.

---

## Seção 6 — Matriz Consolidada: Impacto Esperado, Testes de Regressão e Critérios de Aceitação

| Etapa / Item | Modificação Versionada | Impacto Esperado (Físico / Algorítmico / Métrico) | Testes de Regressão Obrigatórios (`pytest` / `ns-3`) | Critério Quantitativo de Aceitação (`GO`) |
|---|---|---|---|---|
| **Etapa 1** (`v2` isolamento) | Inventário pré-commit + commit documental/`main` + tag local + branch `exp/sqr-v2-infra` (`origin/*` intocado) + `scratch/sqr_v2/`, `src/nsoran/sqr_v2/`, `SQR_V2_MANIFEST.json`, política de seeds | Zero impacto nos dados congelados; preservação integral de `Auditoria.txt` e artefatos preexistentes; rastreabilidade tri-eixo completa | Hash check `B0_FROZEN_MANIFEST.json`; execução de `run_sqr_tests.sh` (`360` testes); checagem `git rev-parse origin/main` | `17/17` hashes intactos; `360 passed`; `origin/*` inalterado; 0 uso de seeds `3001+`/`1201+` |
| **Etapa 2 (`BUG-03`)** (`IPC v2`) | Envelope `sqr_ipc_indication_v2.json` c/ `run_id` + `window_seq`, flush antes do `sem_post`, drain `sem_trywait`, escrita atômica $O(1)$ de ação com `# END_V2`, descarte de anômalas, `observe()` usando estritamente `prb_executed_mac`, `predict_and_uncertainty`, separação `--ipc-mode strict_rt` (`50 ms`) vs `causal_sim` | Elimina corrida de timeout `50 ms` e *shift* permanente de 1 passo; garante reconciliação causal MAC-confirmada; separa prova de `50 ms` de simulação causal | **5 classes de falha sob `50 ms`** (antiga, duplicada, atrasada `>50 ms` isolada/alternada/rajada, incompleta/malformada, outro `run_id`) + **paridade `A/A`** em 2 runs fechados + telemetria `decision_wall_ms` em `strict_rt` | `0` deslocamentos pós-atraso; contadores exatos para as 5 classes de falha; `observe()` usa `prb_executed_mac`; `A/A` 100% bit-idêntica; `strict_rt` vs `causal_sim` explicitado |
| **Etapa 2 (`BUG-05`)** (`C=53` sync) | Sync `total_prb` na leitura de `reset()` via envelope `sqr_ipc_indication_v2.json` com `run_id` | Corrige Passo 1 (`equal` passa a pedir `[18,18,17]=53` já na janela 1) | Teste de Passo 1 em `equal` e `SQR` com `C=53` em `sqr-v2-sim` | `prb_requested(step=1) == [18,18,17]` no `equal`; zero divergência de `C` entre Python e MAC |
| **Etapa 2 (`BUG-04`)** (`Métricas v2`) | `campaign_analysis_v2.py` usando `.first()` p/ `prb_idle_est` e separando `PRBs/slot` vs `RBG·slots/janela`; `sqr_v2_reanalyze_v1.py` | Corrige o erro 3× (`74,96 -> 24,99` no `sqr-g0` `v1`) e elimina comparação inválida entre `205–327` e `0–53` na Fig. 4 sem alterar CSVs brutos | Teste unitário sintético + invariante em reanálise somente-leitura da campanha `v1` em `results/sqr_v2/reanalysis_v1_bug04/` | `rb_cell_total_executed_mean + rb_cell_total_idle_est_mean == C` (`erro < 1e-6`); `scheduled + idle == budget` em `RBG·slots`; 0 alteração nos CSVs brutos `v1` |
| **Etapa 3A (`BUG-01`)** (`MCS`) [*Aguarda Gate 2*] | `SqrMacScheduler`/`RslaqMacScheduler` em `scratch/sqr_v2/` sem `m_dlMcs==0 -> 15` (`--legacyForceMcs15OnZero=false`) | Restaura modulação QPSK legítima (`MCS=0`) em `CQI=1` (`~-2,18 dB`), reduzindo BLER artificial ~100% | **Teste C++ / trace `ue_detail` controlado em `CQI=1`** + checagem `CQI∈{2..15}` e `R27` (i–iv) | `MCS=0` observado >0× em `CQI=1`; throughput `CQI=1` > 0 (`BLER < 1`); `CQI>=2` bit-idêntico; Gate `R27` (i–iv) `PASS` |
| **Etapa 3A (`BUG-06/07/08`)** [*Aguarda Gate 2*] | `wbCqi==0` pós-attach $\to$ `snr_norm = 0.0` (distinto de `-999` pré-attach); guarda `ue.attached`; stop `clientApp` em `UeDeparture`; log `nominal_dl_slots` | Impede mascaramento falso de `15 dB` em outage `CQI=0` e evita linhas KPM pré-RACH / tráfego órfão | Teste unitário `wbCqi=0` vs `-1` + teste C++ de sessão curta (`0,3 s < 0,6 s`) e attach | `snr_norm == 0.0` para `wbCqi=0`; `0` KPMs pré-attach; `0` pacotes após `UeDeparture`; `nominal_dl_slots >= active_dl_slots` |
| **Etapa 3B (`Normativa`)** [*BLOQUEADA*] | Revisão de símbolos DL do slot especial `S` (`P0` vs `P1` 5G-LENA vs 3GPP) e manutenção de `Q0` (`5QI=9`) após retratar erro de `5QI=79` (`NGBR_V2X`) | Garante rigor 3GPP antes de eventual troca de cenário físico/QoS | Somente após desbloqueio explícito do usuário | Aguarda aprovação normativa específica |
| **Etapa 4 (`BUG-02`)** (`L0, L1, L2a, L2b, L3`) | Avaliação sequencial isolada: `L0`, `L1`, **`L2a` (NORMA)**, **`L2b` (Poda)**, `L3` | Isola efeito de regularização vs regra de poda vs escala da ação | Testes unitários de `L2a`/`L2b` + matriz dev pareada | Critério composto: SLA c/ `PASS/FAIL/UNKNOWN`, calibração, contraste $\gamma$, estabilidade pós-$B=2000$, `0` desyncs IPC, `confident > 0%` |
| **Etapa 5 (`Premissa`)** (`E5-Idle` / `E5-Transfer`) | Replays contrafactuais `v2` causais separados (`E5-Idle` e `E5-Transfer`) — **sem novo controlador** | Mede poder de ordenação real de $\hat{f}_\tau$ e $\sigma$ nas condições avaliadas | Paridade `A/A` `v2` nos prefixos de replay + bloqueio de checkpoint futuro | Correlação Spearman $\rho > 0$ e *False Safe Donor Rate* $\le 10\%\text{–}15\%$ em `E5-Transfer`, e ganho limpo em `E5-Idle` (interpretados no escopo avaliado) |

---

## Seção 7 — Auditoria dos Resultados Históricos (`v1`): O que Permanece Válido vs. O que Precisa Ser Reavaliado

| Conjunto de Resultados Históricos (`v1`) | Classificação | Bugs / Fatores Incidentes | Ação Necessária |
|---|---|---|---|
| **Diagnóstico analítico de link-budget 3GPP (`TR 38.901 UMa` @ `2 km / 30 dBm` vs `300 m`)** | ✅ **VÁLIDO (Estrutural)** | Equação física de perda de percurso ($\approx 148,6$ dB NLOS) e déficit de `~25–30 dB` frente a macro BS real (`46 dBm`, `8×8`) não dependem dos bugs | Manter como diagnóstico teórico (citando explicitamente `TR 38.901` vs `TS 36.942` do PDF). |
| **Gate `R27` (i–iv) em `SqrMacScheduler` e paridade IPC×E2 `DEGRADED` (`190646`/`192020`, `3585/3585`)** | ✅ **VÁLIDO (Arquitetural)** | Smoke sem RL complexo (`dt ≈ 10 ms < 50 ms`, `C=53`, `0` mismatches finais) | Manter como prova de conceito de execução MAC absoluta e paridade xApp; no texto declarar escopo `DEGRADED`. |
| **Caracterização estática Rodada 1b e Rodada 2 (`30` runs `equal` × `prop` a `300 m`)** | ✅ **VÁLIDO sob `TDD 7 ms` (`P0`)** | Controladores estáticos (`dt < 1 ms` $\implies$ `0` `BUG-03`), sem regressor quantílico (`0` `BUG-02`), canal `SNR ≥ 20 dB` (`MCS=15` só residual pré-CQI) | **Válido como caracterização sob `P0`.** Apenas testar na **Etapa 3B** uma sensibilidade curta `P0 × P1 (TDD 5 ms)` para isolar o efeito do gap UL (`3 ms -> 2 ms`) na cauda URLLC. |
| **Estudo `G4-B` (`9` seeds inéditas `1201–1209`, `FFR = 11,08%`) e `Emenda 4` (atraso/censura MNAR)** | ✅ **VÁLIDO sob `TDD 7 ms` (`P0`)** | `G4-B` é avaliação offline sem loop IPC/`BUG-03` e sem `va` negativo (`BUG-02`); censura `19,2 KB vs 69 B` é fenômeno físico de entrega | Preservar como evidência do limite sob `P0`; opcionalmente checar impacto de `TDD 5 ms` (`P1`) na etapa `3B`. |
| **Refutação `Gate 1 (A5)` / `Gate Contextual` (`25` estados, `127` replays)** | ✅ **VÁLIDO para `B0-rest v1` (escopo pré-lag)** | Verificamos que todos os `25` estados antecedem os onsets de lag IPC (`1788/2219/2416/2836`) e os `127` replays têm `desync = 0` e paridade `A/A` `25/25` | Preservar integralmente como refutação histórica da heurística `max-min` direta da `A5`; **não** reutilizar como validação confirmatória após correções `v2`. |
| **Tabelas `summary_by_method.csv` e `Fig. 4` da campanha `2 km` (`rb_cell_total_idle_est_mean`, `rb_scheduled_mean`)** | 🔧 **REPROCESSAMENTO ANALÍTICO (`BUG-04`)** | Triplicação de `prb_idle_est` (`74,96`/`76,75`) e mistura de `RBG·slots` com `PRBs/slot` | Rodar `sqr_v2_reanalyze_v1.py` **somente-leitura** gerando tabela errata em `results/sqr_v2/`, sem alterar CSVs brutos. |
| **Campanha base de `40` runs a `2 km` (`sqr-g0`, `sqr-g1`, `kbrl`, `equal`)** | ⚠️ **CONTAMINADA — REAVALIAR NA `V2`** | **`BUG-01`** (`0/68` janelas `CQI=1` com vazão), **`BUG-02`** (`confident = 0%`, $\gamma=0 \times \gamma=1$ nunca atuou), **`BUG-03`** (`18/20` runs SQR dessincronizados) e **`BUG-06`** (`24–35%` de `-999` mascarados) | Embora o colapso de borda a `2 km/30 dBm` persista pela física UMa, **a comparação algorítmica `SQR γ=0 × SQR γ=1 × KBRL` precisa ser refeita na `v2`** (em `300 m` viável e, opcionalmente, em `2 km` corrigido). |
| **Sweep de raio (`300 / 800 / 1500 / 2000 m`) do `README` antigo (`n=1` seed, `240 s`)** | ⚠️ **INCOMPLETO + `BUG-01` — REFAZER NA `V2`** | Apenas `1` seed, CSVs de `1500/2000 m` não persistidos e `BUG-01` matando `CQI=1` em `1500/2000 m` | Re-executar na **Etapa 3A** (`seeds 1..5`, sem `BUG-01`, persistindo CSVs completos em `results/sqr_v2/radius_sweep/`). |
| **Episódios fechados completos de `B0-rest` em `seed 2103` (`lam5_s2103`, `lam6_s2103`) e fim de `lam6_s2102`** | ⚠️ **CONTAMINADO POR `BUG-03` — REAVALIAR NA `V2`** | Verificado via `A5_GATE1_B0_PARITY_REPORT.json` e `sqr_prb_windows.csv`: `lam5_s2103` sofreu desync permanente a partir do passo **`1692`** (`1.263/2.995 = 42,2%` divergentes) e `lam6_s2103` a partir do passo **`1473`** (`1.398/2.995 = 46,7%`), contra `0` em `lam5_s2102` e `97` em `lam6_s2102`! | **Re-executar obrigatoriamente `B0-rest` (`SQR-HET`) em `seeds {2101, 2102, 2103}` sob `IPC v2` (`BUG-03` corrigido)** na **Etapa 4** para medir a variabilidade real entre seeds (`-27,24 p.p.` vs parcela causada pelo timeout IPC). |

---

## Seção 8 — Governança de Autorizações (Atualizada em `v1.1` — `2026-11-17`)

1. **Etapa 1 + Etapa 2:** **`GO` AUTORIZADO EM EXECUÇÃO SEQUENCIAL** (preservando integralmente `v1`, `17/17` hashes do `B0_FROZEN_MANIFEST.json`, *replays*, alterações preexistentes e `origin/*` intocado; parada obrigatória ao fim da Etapa 2 para relatório completo de validação IPC, paridade `A/A`, testes das 5 classes de falha e verificação de hashes).
2. **Etapa 3A:** **`GO APENAS PARA PREPARAÇÃO E TESTES ISOLADOS APÓS APROVAÇÃO DO RELATÓRIO DA ETAPA 2`** — não incorporada automaticamente ao baseline de referência antes da aprovação dos gates anteriores.
3. **Etapa 3B:** **`BLOQUEADA`** (revisão pendente da contabilização de símbolos DL no slot `S` de `P1`; referência inicial mantida em `Q0` `5QI = 9` após correção sobre `5QI = 79` = `NGBR_V2X`).
4. **Etapa 4 e Etapa 5:** **`AGUARDAM GATES ANTERIORES`** — Etapa 4 com separação estrita `L2a (NORMA)` vs `L2b (Poda)` e critério composto (sem usar `% confident` como critério único); Etapa 5 com `E5-Idle` e `E5-Transfer` separados e sem afirmação definitiva global de limitação informacional.
5. **Proibições permanentes nesta janela:** NÃO implementar `SQR-CER`, DRL, *bandit* ou novo controlador; NÃO usar seeds confirmatórias (`{3001..3010}`, `>= 3101`, `{1201..1209}`).
