# Plano de Implementação — SQR (CCNC 2026) no MODA + Integração O-RAN via NORI

> **Versão:** 1.0 — 11/10/2026
> **Base:** auditorias de 3 agentes de pesquisa (artigo SQR, ns-3-dev, ns-o-ran-gym) + análise de viabilidade NORI. Documento operacional complementar ao [`AGENTS.md`](AGENTS.md) (regras de ouro, comandos, contrato de interface) e ao [`GUIA_IMPLEMENTACAO_SQR_O_RAN_PYTHON_NS3.md`](GUIA_IMPLEMENTACAO_SQR_O_RAN_PYTHON_NS3.md) (protocolo científico).
> **Status:** plano aprovado para execução; nenhuma fase iniciada.

---

## 1. Decisões tomadas

| # | Decisão | Justificativa |
|---|---|---|
| D1 | **Caminho direto no ns-3 do MODA** (sem prototipagem Python-only) | A infra difícil já existe e funciona (scheduler slice-aware + IPC Python); o SQR não usa rede neural (cabe em NumPy) |
| D2 | **Primeira campanha = SQR γ=0 + SQR γ=1 + KBRL** (espelho das Figs. 2–4 do artigo) | Reprodução fiel antes de extensões; KBRL é o baseline do próprio artigo |
| D3 | **Cenário primário fiel ao artigo**: 1 gNB, 3 slices eMBB, SLA por UE 0,5/1,0/1,5 Mbps, período 100 ms | Conforme GUIA §5 (reprodução primeiro; heterogeneidade depois) |
| D4 | **NORI como Fase 5** (após o núcleo SQR via IPC funcionar); IPC permanece como fallback | De-risking: resultados core garantidos antes da infra O-RAN (RIC/e2sim); diferencial do artigo de vocês: validação O-RAN que nem o artigo original fez |

## 2. Achados de pesquisa que fundamentam o plano

### 2.1 O artigo SQR — o que especifica e o que falta

**Bibliografia:** Aerman Tuerxun e Akihiro Nakao (Univ. de Tóquio), IEEE CCNC 2026, DOI 10.1109/CCNC65079.2026.11366589. 6 páginas. O **simulador não é identificado** no artigo (nenhuma menção a ns-3/MATLAB/OMNeT) — nossa implementação é **reimplementação fiel**, não reprodução bit a bit.

**Método (SQR — "Safe Quantile Regression"):** MBRL leve **sem rede neural**. Por slice `i`:
- Regressor quantílico kernelizado: `f_τ(x) = Σ_{j=1..B} α_j · k(x, x_j)` com kernel **Matérn**; incerteza `σ_t(x) = 1 − max_j k(x, x_j)`; memória **B = 2000** vetores de suporte com poda adaptativa.
- Atualização online (pinball loss, η=0,1): subpredição `α = η·τ`; sobrepredição `α = η·(τ−1)`; **falha crítica** (ação prevista segura que viola SLA) `α = η·(τ−1)·P_penalty`, **P_penalty=2**.
- Feature: `x_t^(i) = [z_t^(i), a_t^(i)/C]`, com estado `z_t^(i) = [s_t^(i), s̄_t]` (KPIs locais: menor throughput de UE, nº UEs, filas, SNR; global: uso de PRBs das outras slices, total de UEs).
- **Ação**: inteiros `a_t^(i) ∈ {0,…,C}` PRBs por slice, `Σa ≤ C`; se exceder, **projeção proporcional Π** + *atualização virtual* punitiva da proposta original.
- **Política hierárquica**: conjunto seguro `A_safe = {a : f_τ(z,a) ≥ Λ}`; se não vazio → **Modo Confiante** `argmax_a [f_τ(z,a) − γ·(a/C)]`; se vazio → **Modo Incerto**: com ε=0,1 exploração Softmax (T não publicada), senão heurística *Safety-First* k-NN com **K=10**.
- **Margem de segurança**: `a ← a + round(σ·M)`; adaptação dinâmica: mais violações → τ↓ e M↑ (quantis τ∈[0,01; 0,05]; equações do controlador não publicadas).
- Formulação **CMDP**: minimizar uso médio de PRBs sujeito a taxa de violação ≤ δ (exemplo δ=1%); KPI `y_t` = 5º percentil do throughput dos UEs.

**Cenário (Tabela I / Seção V):** 1 macro BS, raio 2 km, 2 GHz, 30 dBm, antena 15 dBi, ruído 9 dB, interferência+ruído −110 dBm, pathloss urban macro (3GPP TS 36.942), fading **Pedestrian A** (TS 36.104), chegadas **Poisson 2 UEs/min**, sessão **exponencial 60 s**, UEs móveis com trajetórias aleatórias por episódio, **3 slices eMBB** com SLA **por UE** 0,5/1,0/1,5 Mbps, meta 99% de SLA, período de decisão **100 TTIs = 100 ms**, atualização do modelo < 100 ms.

**Baselines e resultados:** KBRL (Alcaraz et al., TWC 2022 — SLA por slice, não por UE) × SQR γ=0 × SQR γ=1. Figs. 2–4 (taxa de violação, violação acumulada, RBs usados). Alegações: SQR c/ custo viola só 0,003–0,008% mais; economia de RBs ">60%" (Seção V) vs "20%" (Conclusão) — **inconsistente; não usar como critério de aceite**.

**Lacunas (não publicadas → tudo isso é `[ASSUMIDO]` no registro de premissas):** valor de **C**/largura de banda/numerologia; modelo de tráfego aplicativo (DL/UL, pacotes); velocidade e modelo de mobilidade; nº de episódios/seeds/duração; dimensões exatas e normalização do estado; `T_softmax`; parâmetros do Matérn (ν, ℓ); regra de poda da memória; equações do controlador τ/M; peso exato da atualização virtual; definidor exato de "TTI" (interpretamos slot de 1 ms ⇒ numerologia μ=0).

### 2.2 Auditoria do `MODA/ns-3-dev` — o que já existe pronto

- **ns-3.46-dev** oficial + **5G-LENA v5.2-dev** (`contrib/nr`) **compilado e operacional** (build de 11/10; libs e executáveis presentes). `contrib/nr` tem 3 patches defensivos locais (backups em `MODA/docs/patches_nr/`) — preservar.
- **`RslaqMacScheduler`** (`scratch/rslaq/rslaq-mac-scheduler.{h,cc}`, ~53 KB): meta-scheduler slice-aware que estende `NrMacSchedulerOfdmaRR`; particiona RBGs entre N slices por vetor de pesos; algoritmos intra-slice RR/PF/BCQI; **`SetSliceConfiguration(prbWeights, algos)` reconfigura em runtime** (chamado pelo callback IPC a cada período); `SetSliceUeMapping()` (RNTI→slice). Reutilizável **sem modificação** para o cenário SQR.
- **Loop IPC completo funcionando**: semáforos POSIX nomeados + arquivos `rslaq-kpms.txt` / `rslaq_actions_for_ns3.csv`; timeout C++ de 50 ms por ação com degradação graciosa (mantém config anterior). Detalhes e formatos: `AGENTS.md` §4.
- Cenários existentes: `rslaq-sim.cc` (3,55 GHz, 100 MHz, UEs **estáticos**, slices eMBB/URLLC/MTC, período 10 ms — **diferente do artigo SQR em tudo que importa**) e `GreenRan-slice.cc` (anterior). Nenhum ZMQ/E2/RIC. Sem módulo opengym.
- `run_all_scenarios.sh`: 17 modos baseline × 5 cenários de tráfego (**bug: `NS3_DIR` aponta para caminho antigo — usar override**).

### 2.3 Auditoria do `MODA/ns-o-ran-gym` — DRL existente vs. SQR

- Framework Gymnasium maduro e **agnóstico de agente** (`RslaqEnv.reset()/step()` padrão; qualquer objeto com `act(obs)` funciona). Espaço discreto de 66 ações (simplex passo 0,1) — ideal para um regressor que pontua cada ação candidata.
- DRL existente (para o artigo RSLAQ, **outra linha**): DDQN conv (4×Conv2d+BN+Tanh+FC, replay 128, γ=0,80), SAC double-critic, PPO offline (na verdade bandit contextual), forecaster GRU/LSTM (31→22). Checkpoints em `MODA/models/*.pt`.
- Sinais de segurança prontos: `info["outage_flags"]`, `info["soft_flags"]`, `reward_debug` — insumo para a política hierárquica SQR.
- **Nada SQR existe** (zero Matérn/pinball/quantile/k-NN safety). Lacunas a criar: pacote do agente, loop de treino próprio (clonar `rslaq_train_ddqn.py`), formato de checkpoint (matriz de suporte, sem torch), extensão opcional de `eval_offline_rl_closedloop.py`.
- Orçamento de decisão: 2000 SV × ~53 candidatos × ~16 dims ≈ desprezível em NumPy (limite: 50 ms do semáforo).

### 2.4 Viabilidade NORI — veredito: **sim, com port moderado**

- **NORI** = módulo ns-3 (contrib-style, 12 pares de classes: `E2-interface`, `E2-report`, `kpm-indication`, `ric-control-message`, `oran-interface`, schedulers RL `nr-rl-mac-scheduler-ofdma`/`nr-mac-scheduler-ue-info-rl`…) que implementa a **terminação E2 padronizada O-RAN** dentro do ns-3 via **E2Sim** (fork lasseufpa/Lacava, ASN.1/SCTP, GCC 11), conectando o 5G-LENA a um **Near-RT RIC O-RAN SC Release I** (Kubernetes; Dockerfile e blueprints no wiki). Telemetria por **E2SM-KPM**, controle por **RIC Control (RAN Control SM)**; xApps custom; sem rApps; sem modificação do core ns-3.
- **Pins de versão do NORI**: ns-3-dev @ `ab4cce02` = **ns-3.42** (2024-05-29); 5G-LENA @ `78b7179e` = **2025-02-20** (era v5.x, já com `NrPmSearchFast`).
- **Distância até o MODA**: ns-3.42→3.46 (4 releases de API — o esforço principal do port); NR v5.x→v5.2-dev (**pequena**); scheduler RL do NORI deriva da linhagem GSoC/ns3-gym cujos arquivos (`nr-mac-scheduler-ofdma-ai` etc.) **já existem** no contrib/nr v5.2 do MODA como referência.
- **Maior custo**: infraestrutura operacional (RIC em k8s + toolchain e2sim/ASN1c), não o código. Por isso: Fase 5 isolada, com fallback IPC garantido (go/no-go).

---

## 3. Fases de implementação

### Fase 0 — Contrato de interface + registro de premissas (Agente B; 0,5–1 dia)

**Entregas:**
1. `MODA/docs/sqr/assumptions_register.md` — toda lacuna do §2.1 como `[ASSUMIDO]` com justificativa e impacto.
2. `MODA/docs/sqr/source_traceability.md` — parâmetro → seção/tabela do artigo (modelo do GUIA §2.1).
3. Congelamento do contrato (extensão do vigente — formatos em `AGENTS.md` §4):
   - Ação **inalterada**: `timestamp,sliceId,dedicatedPRB%,minPRB,maxPRB,algorithm`; SQR decide `a_i ∈ {0..C}` e converte `100·a_i/C`; Σa≤C garantido pela projeção **no Python**.
   - KPM estendido por UE: `windowThroughputMbps`, `snrDb`, `queueBytes` (p5/lowest-thr/nº UEs derivam no Python).
4. Premissas radio iniciais (todas `[ASSUMIDO]`): 2 GHz; **numerologia μ=0** (slot 1 ms, coerente com "TTI=1 ms"); **C=52 (10 MHz)** padrão com variante **C=106 (20 MHz)**; tráfego DL UDP greedy calibrado em smoke; mobilidade pedestre ~1 m/s random-waypoint; Matérn ν=1,5; length-scales via normalização das features.

**Aceite:** contrato revisado por A e B; carregador de config falha explicitamente se campo essencial nulo.

### Fase 1 — Cenário ns-3 fiel ao artigo (Agente A; 2–4 dias)

**Entrega:** `MODA/ns-3-dev/scratch/sqr/sqr-sim.cc` (clone adaptado do `rslaq-sim.cc`; **não tocar** em `contrib/nr`, `scratch/rslaq`, `scratch/our_paper`):
- 1 gNB macro, 2 GHz, raio 2 km, **TxPower 30 dBm** (rslaq usa 37), ganho de antena 15 dBi, NF 9 dB, pathloss urban-macro 3GPP, fading aproximando **Pedestrian A** (divergência do 5G-LENA documentada no registro).
- **UEs dinâmicos** (o grande delta): chegadas **Poisson 2 UEs/min** (global; variante por slice na sensibilidade), sessão **exponencial 60 s**, slice aleatório uniforme, criação/remoção limpa de fluxos (RLC UM; patches defensivos já cobrem perdas).
- 3 slices eMBB; **período de decisão 100 ms** (parametrizar o atual 10 ms); SLA por UE 0,5/1,0/1,5 Mbps verificada no Python (y_t = p5 do throughput por UE).
- KPMs com as colunas novas do contrato; log por slice por janela de `prb_requested` (ação pós-projeção), `prb_executed` (pós-conversão %→RBG) e `prb_scheduled` (RBGs reais do MAC — contadores existentes do scheduler).
- Reuso do `RslaqMacScheduler` sem modificação (intra-slice fixo RR ou PF; algoritmo fora do espaço de ação SQR).

**Smoke tests (aceite):** `./ns3 run sqr-sim` com controlador dummy equal-split: churn de UEs funcionando; KPMs fluindo pelo semáforo; throughput por UE coerente com link-budget; `prb_scheduled` ≈ cota; 10 min simulados estáveis; ≥3 seeds sem crash.

### Fase 2 — Agente SQR em Python (Agente B; 2–3 dias)

**Entrega:** `MODA/ns-o-ran-gym/src/nsoran/sqr/` (NumPy/SciPy, sem torch):
- `kernel.py` — Matérn **normalizado** (k(x,x)=1), length-scales por dimensão, vetorizado.
- `quantile_regressor.py` — memória B=2000 com poda documentada (hipótese inicial: descartar vetor de menor influência Σ|α_j|·k); predição `f_τ` e `σ=1−max_j k` (σ=1 com memória vazia); atualização pinball online (η=0,1; regras α do §2.1, incluindo falha crítica ×2).
- `sqr_controller.py` — política hierárquica: `A_safe`; confiável `argmax [f_τ(z,a) − γ·a/C]` (**γ alterna SÓ o custo**); incerto ε=0,1 (softmax T_softmax `[ASSUMIDO]` / k-NN K=10 safety-first com log de qual caminho acionou); margem `a+round(σ·M)`; controlador heurístico de τ,M por janela de violações (regra parametrizada `[ASSUMIDO]`).
- `coordinator.py` — projeção proporcional **inteira** (Σa≤C, arredondamento determinístico) + atualização virtual negativa nas slices cortadas (peso `[ASSUMIDO]`, ablação).
- `sqr_env.py` — ambiente Gymnasium (padrão `NsOranEnv`): obs por slice `[lowest-user-throughput_norm, n_ues_norm, queue_norm, snr_norm]` + global `[PRBs das outras slices, total UEs]`; y_t = p5 thr/UE; período 100 ms.
- `examples/sqr_run.py` — loop fechado clonando a estrutura de `rslaq_train_ddqn.py` (log step_metrics.csv; `ComputeAccounting` para wall-time de decisão); aprendizado **online durante o episódio**; checkpoint próprio = {matriz de suporte, αs, hiperparâmetros, τ/M}.

**Testes (`tests/test_sqr_*.py`, checklist GUIA §3.6):** simetria/normalização do kernel; cobertura de quantil em dados sintéticos estacionários; sinal/magnitude da penalização crítica (subpredição >0, sobrepredição <0, crítica amplificada); orçamento ≤2000 e poda determinística; projeção inteira viável (Σa≤C, 0≤a_i≤proposta); γ=0 vs γ=1 diferem **apenas** no termo de custo; determinismo por seed; bordas (memória vazia, slice sem UE, A_safe vazio, sem vizinhos).

**Gatilho go/no-go:** testes sintéticos passando antes de qualquer run ns-3.

### Fase 3 — KBRL + estáticos (Agente B; 1–2 dias)

- Port do KBRL de `jjalcaraz-upct/network-slicing` como **controlador Python no mesmo ambiente** (mesmas obs/ações); dependências antigas → ambiente isolado se necessário; modificações registradas.
- **Harmonização de granularidade** (KBRL impõe SLA por slice; SQR por UE): logar violações nos dois níveis com denominadores explícitos; métrica primária comum declarada antes dos runs (GUIA §1.1).
- Estáticos de sanidade (equal-split, proporcional-à-demanda) reusando modos do `run_all_scenarios.sh` (com `NS3_DIR` corrigido).

### Fase 4 — Campanha + figuras (Agente B; 2–3 dias)

- **Métricas** (GUIA §11): taxa de violação UE-tempo e slice-tempo; violação acumulada; PRBs requested/executed/scheduled; tempo de decisão p50/p95; frequência/magnitude de projeções; uso dos modos (confiante/incerto); traços τ/M.
- **Figuras equivalentes** às Figs. 2 (taxa de violação vs. meta 1%), 3 (violação acumulada) e 4 (distribuição de RBs) — percentuais recalculados no nosso ambiente.
- **Protocolo:** ≥10 seeds pareadas por método; IC 95% por bootstrap na unidade seed; campanha em script próprio clonando o padrão `run_rslaq_*_paper_*`; sensibilidade: C∈{52, 106}, chegada global vs. por slice.
- Diagnóstico de divergência antes de qualquer conclusão sobre "o artigo estar errado".

### Fase 5 — Integração O-RAN real via NORI (A + B; 3–5 dias + infra)

**5.1 Infra RIC (paralelizável, Agente B):** deploy do Near-RT RIC O-RAN SC Release I (Docker/k8s single-node, blueprints do wiki NORI); build do e2sim (GCC 11, libsctp, ASN1c modificado) — componente C independente do ns-3.

**5.2 Port do módulo NORI (Agente A):** `MODA/ns-3-dev/contrib/nori` — adaptar as 12 classes de ns-3.42→3.46 e NR v5.x→v5.2-dev; reconciliar `nr-rl-mac-scheduler-ofdma` com a linhagem `nr-mac-scheduler-ofdma-ai` já presente; build; smoke `nori-sample` (1 gNB + 1 UE UDP) com **E2-SETUP + RIC Indication** visíveis no RIC.

**5.3 Adaptador de controle (Agente A):** RIC Control (RAN Control SM) → `RslaqMacScheduler::SetSliceConfiguration` — mensagens E2 viram cotas de PRB por slice; KPM indications alimentadas pelos mesmos KPIs do contrato da Fase 0 (paridade semântica com o caminho IPC).

**5.4 SQR como xApp (Agente B):** loop SQR movido para xApp no RIC (KPM via E2, RIC Control a cada 100 ms); reexecutar subconjunto da campanha (carga normal + congestionada, ≥5 seeds) comparando **IPC vs E2**: paridade de resultados + overhead de latência E2E (p50/p95).

**Go/no-go da fase:** se a infra RIC travar, publicar com IPC e reportar o port NORI como validação arquitetural. A campanha IPC completa (Fase 4) é pré-requisito.

---

## 4. Orquestração

**Configuração de modelos** (definida pelo usuário, 11/10/2026): **orquestrador = GLM-5.3** (sessão principal, planeja/revisa/integra e controla go/no-go); **implementadores A e B = GLM-5.3-Flash** (`account:zai-individual-coding-plan/GLM-5.3-Flash`, via workflows dinâmicos com `subagent_model` fixado — ver `AGENTS.md` §7.1).

| Fase | A (C++/ns-3) | B (RL/pesquisa) | Paralelismo |
|---|---|---|---|
| 0 | — | contrato + premissas | B libera contrato; A valida |
| 1 | sqr-sim.cc + KPMs | mock do contrato + esqueleto SQR | total (interface congelada) |
| 2 | suporte a dúvidas de API | agente SQR + testes | total |
| 3 | — | KBRL + estáticos | — |
| 4 | — | campanha + figuras | — |
| 5 | 5.2 + 5.3 | 5.1 + 5.4 | parcial |

**Gatilhos go/no-go (consolidados):** (i) testes sintéticos SQR antes de ns-3; (ii) cota de PRB demonstrável no MAC (`prb_scheduled`) antes de campanha; (iii) campanha IPC completa antes do xApp E2.

## 5. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Lacunas do artigo (C, banda, tráfego, T_softmax, Matérn, poda, τ/M) | assumptions register + sensibilidade; nunca apresentar assumido como publicado |
| Pedestrian A inexistente no 5G-LENA | aproximação documentada + relatório de divergências |
| Churn de UEs (Poisson) é novidade no cenário | smoke runs longos anti-vazamento (RLC/PDCP); patches defensivos já presentes |
| Port NORI ns-3.42→3.46 quebra API no scheduler RL / hooks KPM | linhagem `-ai` v5.2 no MODA como referência; fase isolada com fallback |
| Infra RIC/e2sim (GCC 11, ASN1c, k8s) | 5.1 paralelizável e independente; go/no-go; Dockerfile/blueprints do NORI |
| Inference SQR fora do orçamento de 50 ms | ~2000 SV × 53 candidatos × 16 dims é trivial em NumPy; wall-time logado por passo |
| `run_all_scenarios.sh` com NS3_DIR antigo | sempre `NS3_DIR=$PWD` (regra de ouro §5 do AGENTS.md) |
| Comparação KBRL per-slice vs SQR per-UE enviesada | violações nos dois níveis + métrica primária comum pré-declarada |

## 6. Entregáveis finais

1. `MODA/ns-3-dev/scratch/sqr/sqr-sim.cc` compilado + smoke report.
2. `MODA/ns-o-ran-gym/src/nsoran/sqr/` completo + `tests/test_sqr_*` verdes + `examples/sqr_run.py`.
3. Adaptador KBRL + baselines estáticos rodando no mesmo ambiente.
4. Campanha: scripts + dados (CSV/Parquet) + Figs. 2–4 equivalentes com IC 95% + análise de sensibilidade.
5. Fase 5: `contrib/nori` portado + adaptador RIC Control→SetSliceConfiguration + xApp SQR + relatório IPC vs E2 (paridade + latência).
6. `MODA/docs/sqr/assumptions_register.md` + `source_traceability.md` + relatório de divergências ambiente↔artigo.

## 7. Estimativa

| Bloco | Esforço |
|---|---|
| Fases 0–4 (núcleo SQR via IPC) | 8–13 dias úteis |
| Fase 5 (NORI/E2) | 3–5 dias úteis + setup de infra |
| **Total** | **~11–18 dias úteis** |

---

*Histórico: v1.0 (11/10/2026) — consolidação das auditorias e do plano aprovado; ver `AGENTS.md` para regras operacionais e comando de build/execução.*
