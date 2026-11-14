# sqr_jc_preflight_report.md — First deliverable (§83) da Etapa 2.5 SQR-JC

**Protocolo:** `ETAPA_2_5_SQR_JC_PROTOCOLO_FINAL.md` (congelado 2026-11-03, autoridade máxima §0)
**Data:** 2026-11-03 · **Agente:** B (pre-flight) · **Escopo:** itens 1–12 do §98 (PRE-FLIGHT; PILOT NÃO iniciado)
**Regra de ouro do §0 aplicada:** toda divergência código×despacho×protocolo foi REGISTRADA, nunca corrigida silenciosamente.

---

## 1. SQR audit (G0 — PASS)

`results/sqr_jc_final/00_audit/sqr_original_audit.md` (completo, 17 campos do §3). Síntese: o learner em `src/nsoran/sqr/` é conforme ao artigo onde o artigo publica — **B=2000** (`sqr_controller.py:70`), **τ∈[0.01,0.05]** com τ0=0.05 (`:75-81`), **η=0.1** (`:71`), **ε=0.1** (`:73`), **K=10** (`:74`), **P=2** (`:72`), **σ(x)=1−max_j k(x,x_j)** (`quantile_regressor.py:190-200`), período 100 ms — e `[ASSUMIDO]` registrado onde não publica: ν=1.5 (`kernel.py:5-7`), length_scale=1.0 (`sqr_controller.py:213-214`), poda por influência R10 (`quantile_regressor.py:28-40`), equações da adaptação τ/M R11 (`sqr_controller.py:33-46`), T_softmax=1.0, η_virt=0.05. O SQR-JC herda o learner **sem substituição** (§6): `sqr_jc/kernel_models.py` reutiliza `MaternKernel` e as mesmas regras pinball/crítica/poda/incerteza, com três cabeças (α_safe/α_q25/α_q75) sobre as MESMAS support locations (§9/§10).

## 2. ns-3 / 5G-LENA versions (§18 — FREEZE)

| Campo | Valor | Fonte |
|---|---|---|
| ns3_version | **3.46** (dev) | `MODA/ns-3-dev/VERSION` |
| 5glena_version | **NR-v5.2 (under development)** | `contrib/nr/RELEASE_NOTES.md:18` |
| git_commit_project | `b9f3c0e` (main; árvore de trabalho suja com os artefatos deste preflight — sem commit, conforme regra) | git log |
| git_commit_ns3 | n/a (árvore copiada sem `.git` próprio; rastreada pelo fork Jussi_SBRC) | AGENTS §3.1 |
| git_commit_nr | idem (dentro da árvore acima) | — |
| compiler | **g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0** · cmake 4.2.3 | g++ --version |
| OS | **Ubuntu 26.04.1 LTS**, kernel 7.0.0-42-generic x64 | /etc/os-release |
| Python do projeto | 3.13.12 (venv uv `.venv`) | `tests/run_sqr_tests.sh` header |
| channel helper configuration | `NrChannelHelper::ConfigureFactories("UMa", "Default", "ThreeGpp")`; ShadowingEnabled=true; UpdatePeriod=0 (condição e canal) | `sqr-sim.cc:2451-2454` |
| channel model class | `ThreeGppUmaPropagationLossModel` (TR 38.901 UMa pathloss) + `ThreeGppSpectrumPropagationLossModel` (fading small-scale 3GPP) | `nr-channel-helper.cc:317-319` |
| channel condition class | `ThreeGppUmaChannelConditionModel` (LOS/NLOS 38.901 UMa) | `nr-channel-helper.cc:318-319` |
| grid | `NrHelper` defaults: **NumRbPerRbg=1** (`nr-helper.cc:158-161`), **RbOverhead=0.04** (`nr-helper.cc:155`) → 53 RBs = 53 RBGs | ver item 4 |

**Regra §18 respeitada:** NADA foi atualizado. A versão suporta 3GPP UMa/38.901 → **FREEZE**.

## 3. TR 38.901 UMa mapping (§17)

O cenário é um **single-cell UMa scenario based on the TR 38.901 implementation of the frozen 5G-LENA version**: pathloss/condição/fading da fábrica "UMa" do `NrChannelHelper` (classes acima), gNB h=25 m / UE h=1,5 m (compatíveis com hBS/hUT do 38.901 UMa; `sqr-sim.cc:1989,299`), 2 GHz, 30 dBm, ganho 15 dBi, NF 9 dB, TDD `DL|DL|DL|S|UL|UL|UL` congelado, PF intra-slice, RLC UM, μ=0, 10 MHz. O raio operacional da campanha é **300 m = "calibrated operating point used to isolate resource competition from coverage failure"** (§20; flags congeladas dos runners passam `--cellRadius=300`, como no braço prop da Rodada 2).

**Divergências registradas (não ocultadas):**
1. O artigo SQR usa o perfil de fading "Pedestrian A" (TS 36.104), que a 5G-LENA **não implementa**; o small-scale é 38.901 UMa — divergência já registrada `[ASSUMIDO]` no header do cenário (`sqr-sim.cc:18-24`) e mantida.
2. "UMa 38.901 completo" incluiria layout multi-sítio/19-site — **não afirmar** isso no artigo; a formulação correta é a do §17 acima (single cell).
3. TR 36.942 permanece contexto histórico apenas (§19).

## 4. Resource unit (G1 — PASS)

`results/sqr_jc_final/01_resource_unit/resource_unit_manifest.md`. **C = 53 = 53 RBGs = 53 PRBs** (1 allocation unit = 1 RBG = 1 PRB por slot DL): `m_rbNum = floor(10 MHz×(1−0.04)/180 kHz) = 53` (`nr-phy.cc:584`) × `NumRbPerRbg=1` (default do `NrHelper`, `nr-helper.cc:158-161`) → `total_rbg=53, total_prb=53` **confirmado byte-a-byte na telemetria congelada** (`results_sqr_network_only/seed=*/sqr_prb_windows.csv`). A ação é **cota absoluta por slot DL** (R27): serve ATÉ a cota, ocioso NUNCA é emprestado, cota nunca excedida (`sqr-mac-scheduler.h:19-24`, `.cc:494-498,716-719`).
**Divergência registrada:** o docstring `sqr_env.py:99-100` ("13 RBGs × 4 PRBs + 1") está REFUTADO pelo código (daria 52); mecanismo correto é o acima. Nada alterado; emenda do R01 sugerida ao orquestrador.

## 5. Action semantics (G1 — PASS)

**ACTION_SEMANTICS = ALLOCATED_USAGE** (§22): {a ∈ Z³≥0 : a_i ≤ C, Σa ≤ C} — **27.720** combinações para C=53 (a alternativa BUDGET_PARTITION Σ=C, 1.485, é refutada pelo R27: ocioso real existe e não é redistribuído). Decisão baseada em código, nunca por custo computacional. Invariantes §35 aplicados (`joint_action_space.validate_action`): sem negativos, sem a_i > C, sem violar piso ativo. Enumeração lexicográfica determinística = desempate final (§50/§51/§52).

## 6. SLA manifest (G2 — PASS, com divergência registrada)

`src/nsoran/sqr_jc/sla_manifest.yaml` — extraído do código congelado da Rodada 2 (`oracle/dataset.py:71-73`; `oracle/adjudication.py::_sla_oks`; `oracle/models.py:222`):
- **eMBB** (R36): mediana do `windowThroughputMbps` por UE-janela com demanda da janela governada (stamp (t+1)·100 ms); **≥ 3,5 Mbps** (7 Mbps = fonte DORA); m_e = (T_e−3,5)/3,5.
- **URLLC** (R37): p95 dos atrasos por janela de CHEGADA (32 B @ 1 ms); **≤ 10 ms**; m_u = (10−p95)/10.
- **mMTC** (R38): PDR rolante de 1 s (200 B @ 1000 ms); **≥ 0,95**; m_m = (PDR−0,95)/0,95.
- Vácuo (sem demanda) = satisfeito (semântica frozen do Gate II-d, R19).
- **⚠️ DIVERGÊNCIA REGISTRADA (§0):** o despacho pre-flight pedia "URLLC miss≤5% (o que o Gate II-d usou)". O código congelado do Gate II-d **NÃO usou miss≤5%: usou p95 ≤ 10 ms** (`adjudication.py:404-412`; `models.py:222`); miss_pct é métrica de relatório em toda a Rodada 2, nunca gate. Conforme §7 ("Usar somente uma definição primária, conforme o código auditado"), a primária do SQR-JC é **p95 ≤ 10 ms**; a variante miss (limiar 5%) fica no manifest como **SECUNDÁRIA, report-only** (`SLA_URLLC_MISS_PCT_SECONDARY=5.0`, `sla_margin.py`). Reverter para miss exige decisão do orquestrador e invalida o congelamento.

## 7. Joint feature vector (§11/§12 — implementado e testado)

z_joint (23 dims, só pré-decisão): N_e/u/m; queue_e/u/m; prev_KPI_e (mediana thr), prev_KPI_u (p95 ms), prev_KPI_m (PDR rolante); offered_e/u/m; SNR médio imputado por slice (R09); prev_a_e/u/m; viol_e/u/m (janela fechada); URLLC miss rate + p99. Sem slot para o futuro (§12): `test_no_future_information` + `test_no_future_information_in_builder` PASS. **Ausência documentada:** "pending/retransmission state" mMTC do §11 não é observável no contrato KPM congelado (sem contador RLC) — não inventado; a regra no-demand §42 usa N=0 ∧ queue=0 (documentado em `joint_state.py` e no manifest). x = [z_joint, a/C] (26 dims, §5/§54).

## 8. Normalization (§13 — implementado; congelamento pós-CAL)

Físicas herdadadas/novas: KPI/SLA (÷3,5; ÷10), PDR/miss/viol ∈ [0,1], SNR ÷30 (R09 herdada), alocação ÷C. Sem limite natural (contagens/filas/ofertas): **p1/p99 CAL → clip → [0,1] → congelar** (`feature_scaler.py`; `FeatureScaler.fit` recusa refit quando frozen). Manifest: `feature_scaler_manifest.yaml` gerado pelo CAL (`sqr_jc_calibrate.py`); template do formato em `feature_scaler_manifest.example.yaml`. **Validação offline em run real:** 2.965 janelas CAL de um run prop congelado → manifest produzido sem erro.

## 9. Kernel parameters (herdados + §14)

Matérn ν=1.5, anisotrópico, ls=1.0, k(x,x)=1 — os auditados (G0), reutilizados por importação. Cabeças q25/q75: MESMO kernel/memória/pinball, τ fixos 0.25/0.75, regra crítica só na cabeça safe. Poda: influência da cabeça safe decide o índice UMA vez por update e derruba a MESMA locação nas 3 cabeças (§10; decisão documentada). Grid autorizado §14 (`ls × {0.5,1.0,2.0}`) é a única recalibração pós-CAL, escolhida por menor pinball em validação CAL, depois FREEZE.

## 10. q25/q75 calibration (G4 — mecanismo PASS; resultado ⏳ AGUARDANDO CAL)

Implementado em `examples/sqr_jc_calibrate.py` (§16): distribuição de σ nos pontos CAL; grid de percentis **{50,60,70,80,90,95}**; critério de cobertura P(y≤q25)≈0.25 / P(y≤q75)≈0.75 com **moving-block bootstrap (bloco 5 s = 50 janelas, 1000 reamostras, CI 95%)** + erro absoluto ≤ 0.10 + suporte ≥ 70%; escolhe o MAIOR threshold que passa e congela em `sigma_support_manifest.yaml`; se nenhum passa → **STOP** (sem PILOT).
**Resultado da validação indicativa (1 run real λ=2, 2.965 janelas): STOP** — cobertura q25 ≈ 0.93 vs alvo 0.25 em todos os thresholds (cabeça q25 subprevê na distribuição quase-degenerada de baixa carga). Interpretação honesta: λ=2 isolado não tem variância para calibrar quantis; o CAL real (3 λ × 3 seeds, com overload λ=5/6) decide. Se o STOP persistir no CAL real → §16 manda revisar features/kernel (grid §14) antes do PILOT.

## 11. Support threshold (§15/§16)

σ(x) = 1 − max_j k herdado; OUT_OF_SUPPORT ⇒ MARGINAL, **nunca INFEASIBLE** (§49; `feasibility.classify_slice`). O valor congelado vem do §16 acima (`sigma_support` no manifest); até lá os caminhos conservativos (MARGINAL_UNRESOLVED + fallback de pisos) estão ativos e testados.

## 12. Action-space size (G1/G5)

**27.720** ações (ALLOCATED_USAGE, Σ ≤ 53) — contada por código e por fórmula ((C+1)(C+2)(C+3)/6); 1.485 é o count da partição refutada (documentado). Enumeração vetorial lexicográfica testada (`test_action_semantics`).

## 13. Exhaustive-search benchmark (G5 — harness validado; gate ⏳ AGUARDANDO CAL)

Harness `action_search.benchmark_exhaustive` (§37: ≥100 estados CAL low/mid/high, PASS p95 ≤ 80 ms) **executado sobre um run CAL real reprocessado** (λ=2, memória ~1.079 SV/slice ao fim do stream):
- **Exaustiva: p95 ≈ 12,2 s ≫ 80 ms → FAIL esperado** (12 s por decisão é incompatível com o período de 100 ms).
- **Coarse-to-fine §38** (step 4, refine ±4): ≈ 304 ms — ainda acima de 80 ms com esses parâmetros; os parâmetros são CAL-tunáveis (§38/§80 — ex. step 8/refine menor), e o gate real (§38: concordância ≥98%, regret mediano ≤2%, p95 ≤5%, p95 tempo ≤80 ms sobre 300 estados held-out) decide após o CAL.
- Enquanto o gate não passa, o runtime opera com MARGINAL_UNRESOLVED + fallback conservador (§39) — implementado e testado.
Registro honesto: números indicativos de 1 run λ=2; o benchmark oficial roda nos estados CAL completos (o `run_cal.sh` executa com 100 estados).

## 14. Chosen runtime search

**Exaustiva vetorizada é a referência (§36) e o candidato runtime (§37); dado o achado do item 13, o esperado é o runtime ser coarse-to-fine (§38) com parâmetros congelados no CAL.** A troca de modo é EXPLÍCITA (`JointController.switch_search_mode`; coluna `search_mode` do log §56) — nunca silenciosa. Sem certificado de cobertura, "não encontrei viável" ⇒ MARGINAL_UNRESOLVED (§39), análise na família MARGINAL.

## 15. Physical sanity (G3 — ⏳ AGUARDANDO EXECUÇÃO DO ORQUESTRADOR)

`results/sqr_jc_final/02_radio_sanity/run_radio_sanity.sh` pronto (λ_total=2, seeds 1–3, 300 s, controlador prop com as MESMAS flags congeladas da Rodada 2 — `--slas 3.5,3.5,3.5 --sliceApps=greedy,urllc,mmtc --sliceLambdas=0.667×3 --cellRadius=300 --periodMs 100 --c_prbs 52`; LC_ALL=C, .done idempotente, 3 tentativas, timeout 1800 s, flock, limpeza de semáforos órfãos — lições R56/21-10) + analisador `examples/sqr_jc_radio_sanity.py` (RLF via regex frozen, SNR p5/mediana, distribuição CQI, KPI~0, indicadores PHY → veredicto PASS/STOP §24 em JSON+txt). **Este agente NÃO executou ns-3** (regra do despacho).

## 16. Known blockers

1. **G3/G4/G5 dependem de execução do orquestrador** (scripts prontos; pipeline de calibração validado offline ponta-a-ponta em run real).
2. **URLLC margin — divergência de despacho registrada** (item 6): primária = p95≤10 ms conforme o código auditado; miss≤5% é secundária. Decisão do orquestrador necessária apenas se quiser inverter.
3. **§16 STOP indicativo em λ=2** (item 10): esperado pela falta de variância; se persistir no CAL real, revisar features/kernel (§16/§14) antes do PILOT — sem mascarar com thresholds pós-hoc (§78).
4. **Busca runtime**: exaustiva inviável com memória real (item 13); validar coarse-to-fine no gate §38 (parâmetros CAL-tunáveis). Sem certificado ⇒ MARGINAL_UNRESOLVED (§39).
5. **Narrativa R01 refutada** ("13 RBGs × 4 PRBs + 1"): mecanismo real registrado no G1; emenda do register pendente de ratificação do orquestrador (nenhum código alterado).
6. **pending/retransmission mMTC (§11)** não observável no contrato KPM congelado — feature não inventada; no-demand usa os 2 sinais observáveis.
7. Modo PILOT/MAIN do JointController em loop fechado (`sqr_run.py`) NÃO foi escrito nesta etapa (escopo do preflight = módulos §74 + SQR-HET + testes + scripts); a integração no run loop é o próximo passo de implementação pós-gates, antes do PILOT (§98 item 13).

---

### Estado dos gates (resumo para o §96)

| Gate | Status |
|---|---|
| G0 SQR audit | **PASS** |
| G1 resource semantics | **PASS** |
| G2 SLA semantics | **PASS** (divergência miss/p95 registrada) |
| G3 radio sanity | **PENDENTE-EXECUÇÃO** (orquestrador) |
| G4 kernel calibration | **PENDENTE-EXECUÇÃO** (indicativo: STOP λ=2) |
| G5 search runtime | **PENDENTE-EXECUÇÃO** (indicativo: exaustiva FAIL → §38) |
| G6 baseline validity | **PASS unit-level** (SQR-HET reusa o learner auditado; loop fechado pós-gates) |

**Conforme §98: PILOT NÃO iniciado.** Próximos passos do orquestrador: (1) G3 via `run_radio_sanity.sh`; (2) CAL via `run_cal.sh` (congela scaler/σ_support/q25-q75 e decide G5); (3) re-avaliar `00_audit/preflight_checklist.md`; (4) integrar o JointController/SQR-HET no loop fechado; só então PILOT (2001–2003, λ∈{2,5,6}).

### Artefatos desta etapa (todos novos, sem tocar core/SQR existente)

- Auditoria/manifests: `results/sqr_jc_final/00_audit/{sqr_original_audit.md, preflight_checklist.md}`, `01_resource_unit/resource_unit_manifest.md`, `src/nsoran/sqr_jc/{sla_manifest.yaml, feature_scaler_manifest.example.yaml}`
- Código (§74 + extras): `src/nsoran/sqr_jc/{__init__, sla_margin, yaml_subset, kernel_models, joint_state, feature_scaler, joint_action_space, action_search, feasibility, conditional_floors, joint_controller, online_update, logging, sqr_het}.py`
- Testes (§75+§76): `tests/test_sqr_jc_{kernel, state_scaler, action_space_search, feasibility_floors, controller, regression_round2}.py` — integrados a `tests/run_sqr_tests.sh` (**339 passed**, executado nesta sessão)
- Scripts do orquestrador: `results/sqr_jc_final/02_radio_sanity/run_radio_sanity.sh`, `03_calibration/run_cal.sh` + companions `examples/sqr_jc_{radio_sanity, calibrate}.py`
- Nenhum commit/push realizado (regra do despacho).
