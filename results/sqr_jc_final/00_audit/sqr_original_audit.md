# sqr_original_audit.md — GATE G0 (§3): auditoria da implementação SQR original

**Etapa 2.5 SQR-JC — protocolo FINAL (§0: autoridade máxima; §3: "Esses valores são referência do artigo, não devem ser assumidos como verdade do repositório. Antes de implementar o SQR-JC, o agente deve auditar a implementação atual").**
Data: 2026-11-03 · Agente B · Método: LEITURA DO CÓDIGO FONTE (`src/nsoran/sqr/*.py`), com contraste contra os valores publicados do artigo SQR (referência citada no despacho: B=2000, τ∈[0.01,0.05], η=0.1, ε=0.1, K=10, P=2).
Regra: nenhum valor abaixo foi assumido de memória — cada linha cita arquivo:linha do código congelado lido nesta sessão.

---

## 1. Tabela de auditoria (todos os campos do §3)

| # | Campo | Valor NO CÓDIGO (fonte: linha) | Artigo (referência) | Concorda? |
|---|---|---|---|---|
| 1 | **kernel** | Matérn normalizado, k(x,x)=1, anisotrópico (length-scale por dimensão); formas fechadas ν∈{0.5, 1.5, 2.5, ∞} — `kernel.py:22,48-70,131-147` | "Kernelized Function Approximation / Matérn" (§3) | ✅ (família) |
| 2 | **kernel parameters** | **ν = 1.5** `[ASSUMIDO]` (default do construtor; "The paper does not publish nu"; Fase 0 do plano) — `kernel.py:5-7,76`; **length_scale = 1.0 escalar** (default em `sqr_controller.py:213-214` quando kernel não injetado) | não publicado | ⚠️ `[ASSUMIDO]` — constante congelada KERNEL_NU/KERNEL_LENGTH_SCALE herdada pelo SQR-JC (`kernel_models.py:70-71`) |
| 3 | **feature normalization** | z_i ∈ [0,1]^6 = [lowest_thr/1.5, n_UEs/10, queue/2 MB, snr/30, prb_outros/C, total_UEs/20] (escalas congeladas R09) — `sqr_env.py:76-82,174-211`; feature x = [z, a/C] — `sqr_controller.py:286-290` | não publicado (o artigo não detalha) | ✅ congelado R09; herdado onde aplicável |
| 4 | **tau initial value** | **tau0 = 0.05** (= TAU_MAX) — `sqr_controller.py:81` (TAU_INIT = TAU_MAX), usado como default do construtor `sqr_controller.py:153` | faixa [0.01, 0.05]; valor inicial não publicado | ⚠️ inicial [ASSUMIDO R11] dentro da faixa publicada |
| 5 | **tau min/max** | **[0.01, 0.05]** — `sqr_controller.py:75-76` (TAU_MIN/TAU_MAX), mantido por `set_tau` e pela adaptação (`sqr_controller.py:471-480`; `quantile_regressor.py:158-162`) | [0.01, 0.05] | ✅ |
| 6 | **tau adaptation rule** | Bloco de W=100 períodos (10 s): v_W > 1% ⇒ τ←max(0.01, τ−0.005) e M←min(50, 2M); v_W = 0 por 2 blocos consecutivos ⇒ τ←min(0.05, τ+0.005) e M←max(2, M/2); senão mantém — `sqr_controller.py:33-46,453-483` | comportamento publicado ("mais violações → τ↓, M↑", §IV-C); equações não publicadas | ⚠️ equações `[ASSUMIDO R11]` (comportamento [ARTIGO]) |
| 7 | **M adaptation rule** | mesma regra do bloco acima: M0=10 PRBs, limites [2, 50], dobra sob violação, halva na recuperação — `sqr_controller.py:82-84,240,471-480` | não publicado numericamente | ⚠️ `[ASSUMIDO R11]` |
| 8 | **epsilon** | **ε = 0.1** — `sqr_controller.py:73` (EPSILON), moeda ε no modo incerto (`sqr_controller.py:319`) | 0.1 | ✅ |
| 9 | **eta** | **η = 0.1** — `sqr_controller.py:71` (ETA) → coeficientes pinball em `quantile_regressor.py:57-59,264-272` | 0.1 | ✅ |
| 10 | **K (kNN)** | **K = 10** — `sqr_controller.py:74` (KNN_K); k-NN safety-first: K sucessos mais próximos → argmax f_τ(z, a_j); borda 1 = experiência mais próxima; borda 2 = argmax f_τ — `sqr_controller.py:346-382` | 10 | ✅ |
| 11 | **P_penalty** | **P = 2.0**, ativo nas DUAS variantes γ (γ alterna só o termo de custo) — `sqr_controller.py:72` (P_PENALTY), `quantile_regressor.py:261-266` | 2 | ✅ |
| 12 | **memory size** | **B = 2000** — `sqr_controller.py:70` (MEMORY_SIZE) → `KernelQuantileRegressor(memory_size=2000)` — `quantile_regressor.py:111` | 2000 | ✅ |
| 13 | **memory pruning** | influência I_j = \|α_j\|·k(x_novo, x_j); ao encher, descarta o menor (desempate: índice mais antigo); determinístico, sem RNG — `quantile_regressor.py:28-40,206-214,276-281` | "adaptive pruning" sem regra publicada | ⚠️ regra `[ASSUMIDO R10]` |
| 14 | **uncertainty equation** | **σ(x) = 1 − max_j k(x, x_j)**; 1.0 com memória vazia — `quantile_regressor.py:190-200` | σ = 1 − max k (esperado §15) | ✅ |
| 15 | **decision period** | **100 ms** — contrato §2; `sqr_env.py:288` (periodMs default 100) e cenário `sqr-sim.cc` (KPM period 100 ms) | 100 ms | ✅ |
| 16 | **action unit** | **a_i ∈ {0..C} inteiro; C dinâmico = 53 do telemetry total_rbg** (R01 emendada); conversão dedicatedPRB = 100·a_i/C; unidade auditada no G1: 1 unidade = 1 RBG = 1 PRB por slot DL — `sqr_env.py:93-128,454-487`; ver `01_resource_unit/resource_unit_manifest.md` | artigo: a_i ∈ {0..C}, C = PRBs totais | ✅ (com a qualificação do G1) |
| 17 | **projection rule** | Π proporcional deterministic largest-remainder (exata em inteiros; pass-through se Σ ≤ C — RBGs ociosos permitidos, R27; virtual update negativo R12 no pré-projeção com η_virt=0.05) — `coordinator.py:105-166,49` | "projeção proporcional Π + atualização virtual punitiva" | ✅ (Σ ≤ C + idle é a semântica auditada do R27) |

## 2. Campos adicionais do mecanismo (necessários para o SQR-JC herdar fielmente)

| Campo | Valor no código | Fonte |
|---|---|---|
| Seleção confiante | a* = argmax_{A_safe} [f_τ(z,a) − γ·a/C]; A_safe = {a : f_τ ≥ Λ} | `sqr_controller.py:6-12,307-317` |
| Modo incerto | ε-moeda: softmax sobre f_τ com temperatura T=1.0 `[ASSUMIDO]`; senão k-NN | `sqr_controller.py:87-97,318-328` |
| Margem de segurança | a ← a + round(σ·M), clip [0, C] (APENAS SQR/SQR-HET; **proibida no SQR-JC pós-ação-conjunta**, §45) | `sqr_controller.py:28-29,330-332` |
| Regra crítica | previsto seguro (f_τ ≥ Λ) e y < Λ ⇒ α = η(τ−1)·P | `quantile_regressor.py:261-266` |
| Memória de experiências | ring buffer 2000 de (z, a, y, sucesso) p/ kNN | `sqr_controller.py:90,224-232,440-451` |
| y_t do aprendizado | p5 do windowThroughputMbps por slice (R18); slice sem UE excluída (R19); update com a AÇÃO EXECUTADA (R22) | `sqr_kpis.py:409`; `sqr_run.py:510-514` |
| Checkpoint | support matrix + α + hiperparâmetros + τ/M + janela + experiências + RNG (validação de versão/shape no load) | `sqr_controller.py:490-583` |
| Cenário γ | γ ∈ {0,1} altera SOMENTE o custo na seleção; P=2 sempre (AGENTS §5.7) | `sqr_controller.py:10-13` |

## 3. Divergências e pendências registradas (§0: STOP + registrar, nunca corrigir silencioso)

1. **ν=1.5 e length_scale=1.0 são [ASSUMIDO]** (não publicados); o grid de calibração §14 (`original_lengthscale × {0.5, 1.0, 2.0}`) é a única mudança autorizada pós-CAL — já previsto no protocolo.
2. **Narrativa do R01 inconsistente com o código** (achado do G1, registrado no `resource_unit_manifest.md` §1): "53 = 13 RBGs × 4 PRBs + 1" está REFUTADO; o mecanismo real é `NumRbPerRbg=1` (default do NrHelper, `nr-helper.cc:158-161`) + overhead de RB de 4% (`nr-helper.cc:155`, `nr-phy.cc:584`) → 53 RBGs = 53 PRBs. O número 53 está correto; a explicação no docstring de `sqr_env.py:99-100` não. Nenhum código alterado; emenda do register fica para o orquestrador ratificar.
3. **SLAs do artigo (0,5/1,0/1,5 Mbps, 3 slices eMBB) NÃO são os SLAs da campanha heterogênea** — os congelados R36–R38 (3,5 Mbps / p95≤10 ms / PDR≥95%) estão no `sla_manifest.yaml` (G2), extraídos do código do Gate 2.4-II-d, não do artigo.
4. **Température do softmax T=1.0 e η_virt=0.05 são [ASSUMIDO]** (ablação {0.025, 0.05, 0.1}); entram no SQR-HET como estão (baseline fiel), não no SQR-JC (que não usa softmax nem margem post-hoc).
5. O artigo não publica: regra de poda, equações da adaptação (τ,M), valores iniciais τ0/M0, T softmax, escalas de normalização — todos `[ASSUMIDO]` nos registers R09–R12/R21 e herdados sem alteração.

## 4. Conclusão do G0

A implementação `src/nsoran/sqr/` está **ÍNTEGRA e CONFORME** com o backbone publicado do SQR (B=2000, τ∈[0.01,0.05], η=0.1, ε=0.1, K=10, P=2, σ=1−max k, período 100 ms) e com os registers congelados. O SQR-JC herda o learner **sem modificação** (kernel Matérn, pinball online, memória delimitada com poda R10, incerteza, τ dinâmico — ver `kernel_models.py`, que reutiliza `MaternKernel` por importação). Nenhum bloqueador para G0; a campanha pode congelar parâmetros APÓS a calibração §14/§16 na fase CAL.

**G0 = PASS** (este documento + os 17 campos do §3 preenchidos do código).
