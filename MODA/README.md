# Otimização de Slice RAN via Meta-heurísticas + Baseline DRL (RSLAQ DDQN)

Este repositório contém a infraestrutura experimental completa para otimizar a
distribuição de recursos entre **slices de rede 5G/O-RAN** (eMBB, URLLC, MTC)
usando **meta-heurísticas offline** (GA, PSO, SA e uma híbrida), comparadas com
**schedulers puros** (RR, PF, BCQI), **heurísticas adaptativas** (AQPS e
variantes) e o **baseline DRL RSLAQ (DDQN paper-faithful)**.

O problema é formulação de **otimização contínua**: as variáveis de decisão são
os pesos de alocação dos slices `w = [w_eMBB, w_URLLC, w_MTC]` sobre o simplexo
(`w_i ≥ 0`, `Σ w_i = 1`). Cada candidato é avaliado por uma **simulação ns-3
de ponta a ponta** (5G-LENA, full-stack, FlowMonitor) e pontuado pelo **score
v2 "Path C"** (feasibility-first: restrição rígida de latência de cauda URLLC,
fairness max-min entre slices).

---

## Visão geral dos resultados (5 cenários × 10 seeds)

Score Path C (média sobre 10 seeds; viável = score ≥ 0; maior é melhor):

| Cenário | GA | PSO | SA | Híbrida | RSLAQ DDQN |
|---|---|---|---|---|---|
| low_traffic | 50.3 (9/10 viável) | 50.4 (9/10) | 53.0 (9/10) | 50.5 (9/10) | 45.6 (9/10) |
| normal | −101.8 (9/10) | −89.9 (9/10) | −178.5 (8/10) | −142.4 (8/10) | −223.4 (5/10) |
| congestion | −78.8 (6/10) | −100.8 (5/10) | −98.2 (5/10) | −174.4 (5/10) | **−1871.3 (0/10)** |
| stressed | −480.3 (5/10) | −555.0 (4/10) | −585.2 (4/10) | −283.9 (5/10) | −485.6 (2/10) |
| insufficient_resources | −83.8 (7/10) | −99.0 (7/10) | −176.1 (7/10) | −90.0 (7/10) | **−1096.2 (2/10)** |

Wilcoxon pareado (two-sided, n=10), DDQN vs cada meta-heurística — valores-p:

| Cenário | vs GA | vs PSO | vs SA | vs Híbrida | Leitura |
|---|---|---|---|---|---|
| congestion | **0.027** | **0.004** | **0.020** | **0.027** | DDQN perde de todas |
| insufficient_resources | **0.037** | **0.037** | **0.037** | **0.037** | DDQN perde de todas |
| normal | **0.010** | **0.006** | 0.275 | 0.084 | DDQN perde de GA/PSO |
| low_traffic | 0.285 | 0.238 | 0.152 | 0.285 | empate técnico |
| stressed | 0.557 | 1.000 | 0.922 | 0.492 | empate técnico |

**Conclusão central**: sob o protocolo Path C, as meta-heurísticas offline
superam o baseline DRL paper-faithful com significância estatística nos
cenários de sobrecarga (congestion, insufficient_resources, normal) e empatam
nos demais. A vantagem cresce monotonicamente com a contenção. Otimização
offline e auditável é competitiva onde importa para SLA.

Contra baselines **sem otimização** (comparação pareada por seed, otimização
vs cada baseline; `v2_opt_vs_baseline_paired.csv`): a melhor meta-heurística
vence 10/10 seeds contra todos os 14 baselines em `normal`, `stressed` e
`insufficient_resources`; em `low_traffic` a separação é pequena (vitórias
4/10–9/10, muitos empates) — coerente com regime não saturado.

---

## Camera-Ready WPMC 2026 — Grid Search como referência + seeds independentes (protocolo v1)

Extensão experimental para a versão final do paper de meta-heurísticas:
**Uniform Grid Search** (Δw = 0.05 → 231 pontos) como referência de busca,
**5 seeds ns-3 independentes da calibração** (4-8), função objetivo e
algoritmos congelados, e **instrumentação de custo** (`elapsed_s` por
avaliação, cache nunca contabilizado). Consolidação em
`ns-o-ran-gym/results_controlled/v1_camera_ready/consolidated/`.

### Q1 — Proximidade à referência do Grid (10 pares: Low Traffic + Normal)

| Método | % de J(w) da referência (média) | Faixa |
|---|---|---|
| **GA** | **99,61%** | 96,4–100,5% |
| Híbrida | 99,41% | 96,2–102,2% |
| PSO | 98,95% | 97,1–101,3% |
| SA | 98,48% | 95,0–100,3% |

Em **7/10 pares as meta-heurísticas atingem ≥99%** do melhor ponto da grade;
em **3/10 superam a referência** (política no interior de células da grade).
Com 72 avaliações contra 231 da varredura.

### Q2 — Avaliações até a referência (medianas sobre 10 pares)

| Método | 95% da ref. | 99% da ref. |
|---|---|---|
| Híbrida | **1** | **3** |
| GA | 1,5 | 2 |
| PSO | 2 | 16 |
| SA | 4 | 6 |

### Q3 — Custo da busca (seeds 4-8, Low+Normal; política de concorrência constante)

| Método | Evals | Horas totais | Média s/eval |
|---|---|---|---|
| Grid (referência) | 2.310 | 138,3 | 216 |
| PSO | 720 | 47,6 | 238 |
| GA | 720 | 44,6 | 223 |
| SA | 720 | 43,8 | 219 |
| Híbrida | 840 | 38,9 | **167** |

Meta-heurísticas custam **~⅓ da varredura exaustiva** e atingem a mesma qualidade.

### Q4/Q5 — Seeds independentes: qualidade e SLA

Melhor J(w) por seed (média das 5 seeds independentes): Low Traffic **76,3**,
Normal **86,0**, Congestion (GA completo) **83,60 ± 1,06** — o cenário mais
estável. Pareado contra o melhor baseline por seed: **meta-heurísticas vencem
8/10 pares** em Low+Normal (perdem apenas em Low nas seeds 4 e 8, onde a
variância inter-seed domina). Em Congestion, o GA independente (83,60 ± 1,06)
supera com folga os baselines re-executados (BCQI 69,4 ± 13,4; PF 64,8 ± 12,1;
RR 56,3 ± 4,6).

**Status**: campanha interrompida pelo prazo de submissão — Low Traffic e
Normal **100% completos** (todas as buscas + grade + baselines); Congestion
com GA completo nas 5 seeds (PSO/SA/Híbrida/Grade parciais ou não iniciados;
linha para versão estendida).

---

## MODA — xApp offline destilado das meta-heurísticas (segunda linha de trabalho)

**MODA** (*Metaheuristic-Offline Distilled Allocation*) empacota políticas
neurais treinadas **offline** a partir das trajetórias de busca das
meta-heurísticas como uma xApp conceitual de O-RAN (treinar offline, inferir
online — método inspirado em Bordin et al., CCNC 2025). Três receitas de
destilação sobre **MLPs 14→128→128** (Adam, lr 1e-3, 300 épocas, ~1 s de
treino em CPU):

| Variante | Rede/cabeça | Loss | Rótulo honesto |
|---|---|---|---|
| **MODA-Q** | → 66 saídas (simplex step 0.1) | Regressão-Q (MSE vs score da run) | Bandit contextual Q-regression — **não** é DDQN/TD |
| **MODA-BC** | → 3 + softmax | Cloning do top-20% por cenário | Behavioral cloning supervisado |
| **MODA-RWR** | → mean(3) + softmax | Ponderação por recompensa `exp(R/τ)`, τ=0.1 | Reward-weighted regression |

**Regra de honestidade**: o dataset não tem next-state (avaliações ns-3
independentes), então nenhuma variante usa TD-learning — os labels
`ddqn/sac/ppo` são identificadores de receita, e os checkpoints carregam
`algo_description` com a descrição real.

### Resultados closed-loop (score v1 composto, média ± std, n=3 seeds)

| Cenário | MODA-Q | MODA-BC | MODA-RWR | Best meta. | RSLAQ fixed | Equal |
|---|---|---|---|---|---|---|
| low_traffic | 75,9 ± 30,2 | 74,9 ± 31,8 | **84,1 ± 16,4** | 82,9 ± 18,4 | 83,6 ± 17,3 | 83,7 ± 17,1 |
| normal | **88,3 ± 8,1** | 86,9 ± 10,0 | 87,1 ± 9,6 | 87,3 ± 9,4 | 87,7 ± 9,8 | 87,9 ± 9,9 |
| congestion | 75,6 ± 18,8 | **82,6 ± 5,4** | 54,3 ± 44,4 | 61,8 ± 32,9 | 77,2 ± 8,3 | 75,9 ± 6,0 |
| stressed | 83,0 ± 18,4 | 82,4 ± 14,9 | **83,5 ± 17,0** | 83,6 ± 14,1 | 82,8 ± 18,2 | 84,9 ± 14,8 |

Leituras principais: **MODA-BC domina em congestion** (82,6 ± 5,4 — o melhor
baseline re-otimizado fica em 61,8 ± 32,9); **MODA-RWR é a formulação mais
estável** (menor divergência entre arquiteturas L1 ≤ 0,097 e degradação OOD
mínima); MODA-Q colapsa para um vetor fixo entre cenários e é o mais sensível
à arquitetura — documentado como limitação.

### Robustez OOD e sensibilidade (resposta a peer review)

- **Held-out zero-shot** em `insufficient_resources` (295 Mbps — acima do
  máximo do treino, 245 Mbps): todas as políticas degradam fora do domínio
  (`RESPONSE_PEER_REVIEW_OOD.md`), confirmando que o MODA é **intra-regime** —
  a fronteira de validade é declarada e motiva a linha de controle adaptativo.
- **Ablações**: one-hot vs observáveis (R5a), δ/τ/q (R8), multi-seed de treino
  e grid de arquitetura/lr 81 configs — `ablation_R5a_R8.md`,
  `models/ablation/`.
- **Não-generalização entre seeds** (cross-eval viável 7/10 → 14/90) é o que
  fundamenta a linha seguinte: **xApp DRL offline com transições (s,a,r,s′)**
  (AGENTS.md §14.9).

### Como reproduzir a linha MODA

```bash
cd ns-o-ran-gym
python3 examples/build_offline_dataset.py            # dataset (parquet)
python3 examples/train_offline_rl.py                 # treina MODA-Q/BC/RWR (~1s cada; --seeds p/ multi-seed)
python3 examples/eval_offline_rl_closedloop.py --jobs 6   # malha fechada ns-3 (cache de sidecars)
python3 examples/analyze_closedloop_radio_efficiency.py   # eficiência por slice
python3 examples/generate_moda_paper_artifacts.py    # figuras + tabelas IEEE
```

Artefatos: `models/{ddqn,sac,ppo}_offline.pt`,
`models/closedloop_eval/` (72 células), `paper_moda_offline_drl/` (figuras,
tabelas, docs de resposta a revisão).

---

## Estrutura do repositório

```
artigo_jussi/
├── ns-3-dev/                    # simulador ns-3 (5G-LENA) + scratch/rslaq/
│   ├── scratch/rslaq/
│   │   ├── rslaq-sim.cc         # topologia, cenários, CLI, métricas/summary
│   │   └── rslaq-mac-scheduler.cc  # scheduler MAC slice-aware (RBG por peso)
│   └── contrib/nr/              # módulo 5G-LENA (gitignored; patches em docs/)
├── ns-o-ran-gym/                # pipeline Python (meta-heurísticas, DRL, análise)
│   ├── examples/                # runners e scripts de campanha/análise
│   ├── src/nsoran/scoring.py    # função objetivo (score v1 e v2/Path C)
│   └── src/environments/        # Gymnasium env RSLAQ (IPC semáforo+CSV)
├── paper_v2_campaign/           # resultados consolidados v2 + figuras
├── paper_moda_offline_drl/      # segunda linha (offline DRL "MODA")
├── models/                      # checkpoints + resultados closed-loop MODA
├── docs/patches_nr/             # cópias dos patches 5G-LENA (ver abaixo)
└── AGENTS.md                    # escopo da pesquisa + estado experimental
```

### Cenários e configuração de UEs por slice (`InitScenarios()` em `rslaq-sim.cc`)

| Cenário | UEs (eMBB/URLLC/MTC) | Total | Taxa por slice (eMBB/URLLC/MTC) | Pacotes (eMBB/URLLC/MTC) | Oferta total | Regime |
|---|---|---|---|---|---|---|
| low_traffic | 2/2/6 | 10 | 55 / 1 / 1 Mbps | 1500/50/100 B | 57 Mbps | leve |
| normal | 5/5/10 | 20 | 70 / 1 / 2 Mbps | 1500/50/100 B | 73 Mbps | moderado |
| congestion | 15/10/35 | 60 | 180 / 5 / 60 Mbps | 1500/100/500 B | 245 Mbps | saturado |
| stressed | 8/12/20 | 40 | 100 / 3 / 25 Mbps | 1500/100/500 B | 128 Mbps | degradado |
| insufficient_resources | 20/10/40 | 70 | 220 / 5 / 70 Mbps | 1500/100/500 B | 295 Mbps | sobrecarga extrema |

**Mecânica por slice:**

- A **taxa do slice é dividida igualmente entre seus UEs** (ex.: congestion = 15 UEs eMBB × 12 Mbps cada). Cada UE recebe 1 fluxo UDP downlink próprio (porta única mapeia UE→slice para o FlowMonitor atribuir métricas por slice).
- **Atribuição por faixa contígua de ID**: UEs 1..N_eMBB = eMBB, seguintes N_URLLC = URLLC, restantes = MTC (`GetSliceForUe`).
- **Perfil por slice**: eMBB = poucos UEs, pacotes de 1500 B, alto throughput por UE; URLLC = pacotes de 50-100 B, alvo de delay p99 ≤ 10 ms (gate rígido do Path C); MTC = muitos UEs, pacotes de 100-500 B, proteção anti-starvation.
- **Posições**: disco uniforme de 100 m ao redor do gNB, UEs estáticos — a seed do ns-3 sorteia as posições (principal fonte de variância entre seeds).
- **Rádio comum**: 100 MHz @ 3,55 GHz, numerologia 1, TDD `D|D|8D|4GB|4U|U|U`, ~266 RBs = **16 RBGs** particionados pelos pesos otimizados; alocação intra-slice por PF.
- Tráfego em taxa constante (sem rajadas intra-run) — ver limitações no AGENTS.md §14.4.

---

## Instalação e build

Pré-requisitos: Linux, CMake ≥3.13, GCC ≥9, Python ≥3.10 com `torch`,
`numpy`, `pandas`, `scipy`, `matplotlib`, `posix_ipc`, `gymnasium`.

```bash
# 1. ns-3 (C++)
cd ns-3-dev
./ns3 configure --enable-examples --enable-tests
./ns3 build rslaq-sim -j 10          # alvo usado por toda a campanha

# 2. Python (ns-o-ran-gym)
cd ../ns-o-ran-gym
python3 -m pip install -e .          # ou usar sys.path como os runners fazem
```

### Patches defensivos 5G-LENA (OBRIGATÓRIOS)

O módulo `contrib/nr` não é rastreado pelo git (subtree externo). Aplicar as
cópias patchadas antes de rodar campanhas longas — sem elas, perdas pesadas em
RLC-UM abortam a simulação (SIGABRT) em certas seeds:

```bash
cp docs/patches_nr/nr-pdcp.cc      ns-3-dev/contrib/nr/model/
cp docs/patches_nr/nr-rlc-um.cc    ns-3-dev/contrib/nr/model/
cp docs/patches_nr/nr-net-device.cc ns-3-dev/contrib/nr/model/
cd ns-3-dev && ./ns3 build rslaq-sim
```

Os 4 patches (3 em `contrib/nr` + 1 já versionado em
`scratch/rslaq/rslaq-mac-scheduler.cc`) descartam PDUs/SDUs malformadas com
`NS_LOG_WARN` em vez de abortar, e normalizam pesos com drift de arredondamento:
1. `NrPdcp::DoReceivePdu` — valida bit D/C e tamanho antes de `RemoveHeader`.
2. `NrRlcUm::ReassembleAndDeliver` (WAITING_SI_SF) — descarta PDU com FI inesperado.
3. `NrNetDevice::Receive` — exige tamanho mínimo antes de `PeekHeader<Ipv4Header>`.
4. `RslaqMacScheduler::SetSliceConfiguration` — renormaliza `Σw=1` (tolerância a 1e-6 de drift).

---

## Como rodar

### 1. Baselines (schedulers puros, AQPS, heurísticas) — 14 modos

```bash
cd ns-o-ran-gym
REPO_ROOT=$(cd .. && pwd) RUN_TAG=baselines_demo \
OUTPUT_ROOT=$PWD/results_controlled/demo/baselines \
SCENARIOS="normal" SEEDS="1 2 3" SIM_TIME=5 \
RUN_BASELINES=1 RUN_METAHEURISTICS=0 RUN_META_EVALUATION=0 RUN_RSLAQ_DDQN_PAPER=0 \
bash examples/run_all_scenarios.sh
```

Modos: `pure_rr pure_pf pure_bcqi slice_weighted_{pf,rr,bcqi} psta_equal
slice_aqps slice_demand_greedy slice_sla_greedy slice_least_waste
slice_qos_mixed slice_random_vine slice_meta_risk_elastic`.

### 2. Meta-heurísticas (busca offline por cenário×seed)

```bash
cd ns-o-ran-gym
REPO_ROOT=$(cd .. && pwd) RUN_TAG=v2_demo \
OUTPUT_ROOT=$PWD/results_controlled/demo/meta \
SCENARIOS="normal" SEEDS="1 2 3 4 5 6 7 8 9 10" SIM_TIME=5 \
META_ITERATIONS=8 META_POPULATION=6 META_MUTATION_STRENGTH=0.12 \
META_RANDOM_SEED=2026 META_INTRA_ALGO=PF META_SCORE_VERSION=v2 META_PER_SEED_SEARCH=1 \
RUN_BASELINES=0 RUN_METAHEURISTICS=1 RUN_META_EVALUATION=0 RUN_RSLAQ_DDQN_PAPER=0 \
bash examples/run_all_scenarios.sh
```

Cada avaliação = 1 simulação ns-3 standalone (`--baselineMode=slice_custom
--weights=w0,w1,w2 --intraAlgo=PF`). Orçamento por (cenário, seed):
**100 avaliações** (GA 24 + PSO 24 + SA 24 + híbrida 28). Cache incremental:
cada eval grava `candidate.json`; restarts pulam avaliações idênticas
(tolerância 1e-9) e respeitam a versão do score.

### 3. RSLAQ DDQN paper-faithful (baseline DRL)

Treino online via IPC (semáforo POSIX + CSV, 1 ação por frame de 10 ms),
fiel a Yungaicela-Naula et al. (IEEE TMC 2026), Hyp-set3 (Table VI):
`LR=0.001, γ=0.80, λϵ=0.998, ϵ_min=0.05, L=500, btsz=350, nsut=200, ntsr=100,
E≈3500 steps (35 episódios × 100), psta=0.5, ω=[0.3333,0.4000,0.2667]`,
reward piecewise Eq. 12, QNetwork 4×Conv2D+BN+Tanh+FC, 198 ações discretas
(66 pesos × 3 schedulers).

```bash
cd ns-o-ran-gym
bash examples/run_rslaq_ddqn_paper_20260811.sh          # low_traffic+normal+stressed, seeds 1-5
bash examples/run_rslaq_ddqn_paper_seeds6to10.sh        # mesmos cenários, seeds 6-10
bash examples/run_rslaq_ddqn_paper_congestion.sh        # congestion seeds 1-5
bash examples/run_rslaq_ddqn_congestion_seeds6to10.sh   # congestion seeds 6-10
bash examples/run_rslaq_ddqn_paper_insufficient.sh      # insufficient seeds 1-5
bash examples/run_rslaq_ddqn_insufficient_seeds6to10.sh # insufficient seeds 6-10
```

Todos gravam na mesma árvore
`results_controlled/heuristics_metaheuristics/20260811_rslaq_ddqn_paper/rslaq_ddqn_paper/`
(`ddqn_paper_<cenário>_seed<N>/` com `ddqn_best.pt`, `ddqn_training_log.csv`,
`ddqn_summary.json`). Paralelismo: `DDQN_DRL_JOBS=5`.

### 4. Pipeline pós-treino do DDQN (métrica comparável)

O reward paper (escala própria) não é comparável ao score Path C. O pipeline
extrai a política greedy, re-executa ns-3 standalone e pontua igual às
meta-heurísticas:

```bash
cd ns-o-ran-gym
OUT=results_controlled/heuristics_metaheuristics/20260811_rslaq_ddqn_paper/rslaq_ddqn_paper

# a) extrai pesos médios (ação greedy ε=0) por (cenário, seed)
python3 examples/extract_ddqn_weights.py \
    --ddqn-root "$OUT" --output "$OUT/ddqn_extracted_weights_ci.csv" \
    --ns3-dir ../ns-3-dev --eval-episodes 3 \
    --scenarios "congestion,insufficient_resources"

# b) re-executa standalone e pontua com Path C (workers paralelos por cenário)
python3 examples/rescore_ddqn_standalone.py \
    --weights-csv "$OUT/ddqn_extracted_weights_ci.csv" --ns3-dir ../ns-3-dev \
    --output-root "$OUT/ddqn_rescored" --output-csv "$OUT/ddqn_scored_v2_congestion.csv" \
    --sim-time 5 --scenarios "congestion" &

# c) análise unificada + Wilcoxon n=10 + boxplot
python3 examples/analyze_ddqn_vs_meta.py \
    --meta-csv ../paper_v2_campaign/v2_meta_unified_5scenarios.csv \
    --ddqn-csv "$OUT/ddqn_scored_v2_all.csv" \
    --baseline-csv results_controlled/heuristics_metaheuristics_v2/v2_opt_vs_baseline_paired.csv \
    --output-dir ../paper_v2_campaign/
```

### 5. Análises v2 (meta-heurísticas vs baselines)

```bash
python3 examples/analyze_v2_stats.py --meta-root <metaheuristics_root>   # Friedman/Wilcoxon
python3 examples/analyze_v2_opt_vs_baseline.py \
    --meta-root <metaheuristics_root> --baseline-root <heuristics_ns3_root>
python3 examples/generate_v2_audit_figures.py --meta-root <metaheuristics_root>
```

---

## As meta-heurísticas em detalhe

Todas otimizam o mesmo objeto: o vetor de pesos `w ∈ R³` no simplexo
(`w_i ≥ 0`, `Σ w_i = 1`), mapeado no scheduler MAC para orçamentos inteiros de
RBGs por slice (`⌊totalRbgs·w_s/Σw_ativos⌋` + redistribuição de resto por
largest-remainder), com alocação intra-slice por PF. Hiperparâmetros comuns da
campanha v2: `population=6`, `iterations=8`, `mutation_strength=0.12`,
`random_seed=2026` (RNG recriado por cenário×seed e consumido na ordem
GA→PSO→SA→híbrida), score v2 (Path C), 10 seeds por cenário.

### GA — Algoritmo Genético (24 avaliações/par)

- **Representação**: vetor contínuo 3-D no simplexo; viabilidade garantida por
  renormalização após cada operador.
- **Inicialização**: 6 indivíduos dos priors guiados (`seed_weight_candidates`:
  vetores baseados em conhecimento do domínio, ex. eMBB-heavy, URLLC-heavy,
  uniforme) + amostragem aleatória.
- **Seleção**: elitista — o melhor indivíduo é preservado entre gerações.
- **Crossover**: BLX-α (`crossover_weights`): filho = α·pai1 + (1−α)·pai2 por
  coordenada, α ~ U(0,1), seguido de renormalização.
- **Mutação**: perturbação gaussiana por coordenada (`w_i + N(0, 0.12)`) com
  clip para `w_i ≥ 0` e renormalização (`mutate_weights`).
- **Orçamento**: 24 avaliações (população inicial + gerações dentro de
  `iterations=8`).

### PSO — Particle Swarm Optimization (24 avaliações/par)

- **Enxame**: 6 partículas com posição (pesos) e velocidade 3-D.
- **Atualização**: velocidade clássica (inércia + atração ao pbest + gbest);
  posição atualizada por soma e **projetada de volta ao simplexo por
  renormalização** a cada passo.
- **Limitação documentada**: a projeção pós-passo destrói a semântica canônica
  do vetor velocidade (limitação reconhecida no paper; funciona
  empiricamente, mas não é PSO estritamente canônico).
- **Orçamento**: 24 avaliações, mesmo critério de parada do GA (justiça de
  custo computacional).

### SA — Simulated Annealing (24 avaliações/par, orçamento equalizado)

- **Trajetória única**: começa na vizinhança do centro do simplexo
  ([1/3,1/3,1/3]); sem população.
- **Vizinhança**: mesma mutação gaussiana σ=0.12 + renormalização.
- **Aceitação**: Metropolis — aceita piora com probabilidade exp(−Δ/T);
  a escala de temperatura usa fator de compensação ad-hoc (documentado como
  limitação) para a magnitude do score.
- **Orçamento equalizado**: 24 avaliações, idêntico a GA/PSO (historicamente o
  SA fazia apenas `iterations` evals — busca por trajetória única — o que era
  comparação injusta; a equalização resolve isso).

### Meta-heurística híbrida (28 avaliações/par)

- **Estrutura**: memética — operadores de população (exploração global)
  combinados com refinamento local tipo SA (exploração local) a cada iteração.
- **Componentes**: mesma inicialização guiada do GA + BLX-α + mutação gaussiana
  (global); passo de busca local Metropolis aplicado ao melhor indivíduo de
  cada iteração (local).
- **Orçamento**: 28 avaliações (24 do ciclo populacional + 4 do refinamento
  local). A comparação primária usa o teto comum de 24; as avaliações extras
  (efeito do refinamento) são reportadas separadamente.
- **Status no paper**: avaliação comparativa — **não** é claim de superioridade
  (perde do PSO puro em alguns cenários; Friedman p=0.29 no piloto congestion).

### Função objetivo — score v2 "Path C" (`scoring.py::score_summary_rows_v2`)

```
restrição rígida: p99_URLLC ≤ 10 ms (por pacote; URLLC morto → sentinela 10 s)
inviável  → score < 0, proporcional ao excesso de latência (gradiente à viabilidade)
viável    → 100 · (0.5·min_sla + 0.5·mean_sla)   [fairness max-min pune starvation]
```

onde `sla_i = min(S_thr, S_pdr, S_delay)` por slice (satisfação composta,
alvos por slice definidos no C++). O score v1 (campanha original, soma
ponderada com penalidades) permanece implementado e selecionável via
`--score_version v1`; **nunca misturar escalas v1/v2 sem rótulo**.

---

## Saídas

```
results_controlled/<tag>/
├── heuristics_ns3/results_rslaq_network_only/
│   └── scenario=<s>/mode=<m>/seed=<n>_run=1/{summary.csv,timeseries.csv,...}
├── metaheuristics/
│   └── scenario=<s>/seed=<n>/metaheuristic_search/
│       ├── evals/<metodo>_eval_NNNN/      # 1 sim por candidato + candidate.json (cache)
│       ├── metaheuristic_results_<s>_seed<n>.csv   # trajetória de convergência
│       └── best_candidate_<s>_seed<n>.json
├── rslaq_ddqn_paper/ddqn_paper_<s>_seed<n>/
│   ├── ddqn_best.pt / ddqn_final.pt / ddqn_training_log.csv / ddqn_summary.json
│   └── <uuid>/                             # 1 dir por episódio (IPC)
└── ddqn_rescored/scenario=<s>/mode=rslaq_ddqn/seed=<n>_run=1/summary.csv
```

`summary.csv` (por slice): throughput, delay mean/p95/p99/p999 (por pacote),
PDR, PLR, buffers, satisfação SLA composta, utilização de orçamento, violação
de deadline, entre outros.

## Limitações (resumo)

- Nomes de cenários são perfis de entrada; a viabilidade é fortemente
  dependente da seed (a variância entre seeds domina a diferença entre
  métodos fora do congestionamento).
- O DDQN é treinado com o reward paper (Eq. 8/12), não com Path C; a comparação
  mede a transferência reward→KPIs (pesos extraídos re-executados em condições
  idênticas). Declarar explicitamente no paper.
- Pesos ótimos não generalizam entre seeds (cross-eval viável ~11–16%) —
  otimização offline de pesos estáticos é insuficiente para deployment;
  motiva adaptação online.
- Hiperparâmetros modestos (pop=6, iter=8) por custo de simulação;
  estudar ablação antes de ampliar.

Consulte `AGENTS.md` para o estado experimental completo, bugs conhecidos e
regras de escopo.
