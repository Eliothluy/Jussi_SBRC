# AGENTS.md — Guia de Operação para Agentes

> Projeto: reprodução/estensão do artigo **"Safe RAN Slicing in O-RAN: Minimizing SLA Violations via Model-Based Reinforcement Learning"** (SQR, IEEE CCNC 2026) no ambiente **MODA** (ns-3/5G-LENA + Python RL), com integração do protocolo O-RAN real (E2/RIC) via **NORI** em fase final.
>
> Este arquivo é a **fonte única de verdade operacional** para qualquer agente (humano ou IA) que trabalhe neste workspace. O plano detalhado de implementação está em [`PLANO_IMPLEMENTACAO_SQR_NORI.md`](PLANO_IMPLEMENTACAO_SQR_NORI.md).

---

## 1. Objetivo do projeto

1. **Reimplementar o método SQR** (regressão quantílica online segura com kernel Matérn, **sem rede neural**) do artigo CCNC 2026 (Tuerxun & Nakao, Univ. de Tóquio, DOI [10.1109/CCNC65079.2026.11366589](https://doi.org/10.1109/CCNC65079.2026.11366589)) sobre a plataforma MODA já existente.
2. **Campanha experimental espelhando o artigo**: SQR γ=0 × SQR γ=1 × KBRL (Figs. 2–4: taxa de violação de SLA, violação acumulada, uso de RBs).
3. **Fase final (diferencial)**: integrar o protocolo O-RAN real via **NORI** ([lasseufpa/nori](https://github.com/lasseufpa/nori)) — interface E2 padronizada + Near-RT RIC — com o agente SQR rodando como **xApp**.

**Decisões já tomadas** (não reabrir sem motivo forte):
- Caminho de implementação: **direto no ns-3 do MODA** (sem fase de prototipagem Python-only), reaproveitando o IPC semáforos/CSV existente.
- Cenário primário: **fiel ao artigo** (1 gNB, 3 slices eMBB, SLA por UE 0,5/1,0/1,5 Mbps, período de decisão 100 ms).
- NORI entra **após** o núcleo SQR funcionar via IPC; o IPC permanece como fallback e caminho de debug.

## 2. Documentos-fonte no workspace

| Arquivo | Papel |
|---|---|
| [`tradução.md`](tradução.md) | Tradução do artigo SQR (verificada contra o PDF de 6 páginas) |
| `Safe RAN Slicing in O-RAN: Minimizing SLA
Violations via Model-Based Reinforcement Learning.pdf` | Artigo original (**atenção: o nome do arquivo contém quebra de linha entre "SLA" e "Violations"**) |
| [`GUIA_IMPLEMENTACAO_SQR_O_RAN_PYTHON_NS3.md`](GUIA_IMPLEMENTACAO_SQR_O_RAN_PYTHON_NS3.md) | Protocolo de pesquisa completo do grupo (rastreabilidade, fases E0–E9, protocolo estatístico, perguntas aos autores) — **obrigatório consultar antes de decisões metodológicas** |
| [`PLANO_IMPLEMENTACAO_SQR_NORI.md`](PLANO_IMPLEMENTACAO_SQR_NORI.md) | Plano de implementação detalhado (Fases 0–5) com os achados de pesquisa que o fundamentam |

## 3. Mapa do ambiente (caminhos exatos)

```
/home/eliothluy/Documentos/Jussi_SBRC/        ← raiz deste workspace (git repo, branch main)
├── AGENTS.md                                  ← este arquivo
├── PLANO_IMPLEMENTACAO_SQR_NORI.md            ← plano detalhado
├── tradução.md / PDF do artigo / GUIA...
└── MODA/                                      ← repo git próprio (commit único 3d18f43)
    ├── ns-3-dev/                              ← ns-3.46-dev + 5G-LENA v5.2-dev — JÁ COMPILADO
    │   ├── contrib/nr/                        ← 5G-LENA v5.2-dev (RELEASE_NOTES: "NR-v5.2 under development")
    │   │   └── ⚠️ contém 4 PATCHES DEFENSIVOS locais (ver §3.1)
    │   ├── scratch/rslaq/                     ← cenário + scheduler slice-aware atuais (NÃO MODIFICAR)
    │   │   ├── rslaq-sim.cc                   ← cenário RSLAQ (3,55 GHz, UEs estáticos, eMBB/URLLC/MTC, 10 ms)
    │   │   └── rslaq-mac-scheduler.{h,cc}     ← RslaqMacScheduler (partição de RBGs por slice, runtime)
    │   ├── scratch/our_paper/                 ← cenário GreenRAN (versão anterior; scheduler com SetSliceConfiguration)
    │   ├── scratch/sqr/                       ← (A CRIAR — Fase 1: cenário fiel ao artigo SQR)
    │   └── build/                             ← libs libns3.46-*-default.so + executáveis prontos
    ├── ns-o-ran-gym/                          ← pacote Python (fork ns-O-RAN Gym, Lacava et al.)
    │   ├── src/nsoran/                        ← core (ns_env, action_controller, datalake, offline_models)
    │   ├── src/environments/                  ← rslaq_env, rslaq_reward, rslaq_kpis, rslaq_action_spaces...
    │   │   └── src/nsoran/sqr/                ← (A CRIAR — Fase 2: agente SQR)
    │   ├── examples/                          ← scripts de treino/eval (rslaq_train_ddqn/sac etc.)
    │   ├── tests/                             ← pytest (15+ arquivos)
    │   └── run_training.sh
    ├── models/                                ← checkpoints DDQN/SAC/PPO (.pt) + avaliações closed-loop
    └── docs/patches_nr/                       ← backup dos 3 patches do contrib/nr
```

### 3.1 Estado do ns-3-dev (auditoria 11/10/2026)

- **Versão**: ns-3.46-dev (`VERSION`); árvore oficial `nsnam/ns-3-dev` copiada (sem `.git` próprio) dentro do repo MODA.
- **5G-LENA**: v5.2-dev em `contrib/nr`; rastreado apenas pelo repo **Jussi_SBRC** (fork local — o repo `Eliothluy/MODA` permanece intocado) e recebeu **4 patches defensivos**: `nr-pdcp.cc`, `nr-rlc-um.cc`, `nr-net-device.cc` (convertem `NS_ASSERT` em warn/drop para perdas pesadas de RLC-UM não abortarem a simulação) e `nr-qos-rule-classifier.cc` (dedup de regras QoS idênticas reenviadas pelo MME a cada reestabelecimento RLF — sem ele, runs longos crasham no assert ≤64; autorizado pelo orquestrador em 12/10/2026). Backups em `MODA/docs/patches_nr/`. **Nunca reverter/atualizar contrib/nr sem preservar esses patches.**
- **Build íntegro**: `build/lib/` completo (incl. `libns3.46-nr-default.so`); executáveis `ns3.46-rslaq-sim-default` e `ns3.46-GreenRan-slice-default` prontos.
- **⚠️ Atenção Python (descoberto 11/10/2026)**: o wrapper `./ns3` **quebra com o Python 3.14 do sistema** (`argparse` do 3.14 rejeita `store_true` em argumento posicional — ValueError no `parse_args`). Sempre invoque o ns-3 via `bash MODA/ns-3-dev/scratch/sqr/ns3_gate.sh <args>` (wrapper que usa o **Python 3.13 do uv** em `~/.local/share/uv/python/cpython-3.13-*/bin/python3.13`; também há 3.11 disponível). Interpreters uv são a alternativa quando o 3.14 do sistema causar problemas.
- **Sem** módulo opengym (`contrib/` só tem `nr`) — os schedulers RL `nr-mac-scheduler-*-ai` do 5G-LENA existem no código mas ficam descompilados sem `contrib/opengym`. **Sem** ZMQ, **sem** E2/RIC (hoje).
- `src/` tem 43 módulos padrão; nada de `oran`/`e2`/`gym`.

### 3.2 Estado do ns-o-ran-gym (auditoria 11/10/2026)

- Python ≥3.12; deps: `gymnasium>=0.29`, `posix-ipc`, `pandas`, `numpy>=2.1.1`, `sem`; extra `[ml]`: `torch>=2.2` (opcional).
- `RslaqEnv` (Gymnasium): obs `Box(4,4)` (btx, bfs, rsh, tdp × eMBB/URLLC/MTC/célula); ações contínuas `Box(3,)`→softmax→%PRB ou `Discrete(66)` simplex passo 0,1 (198 c/ scheduler RR/PF/BCQI).
- DRL existente: DDQN (conv nets, replay 128, γ=0,80) e SAC online; DDQN/SAC/PPO "offline" são bandits contextuais PyTorch (`offline_models.py`); forecaster GRU/LSTM (`rslaq_predictive.py`, FEATURE_DIM 31/FORECAST_DIM 22).
- **Nada SQR existe**: zero Matérn/pinball/regressão quantílica/k-NN safety-first/ε-softmax no repo inteiro. O agente SQR será um **novo consumidor do ambiente** (sem tocar no core).
- `REPORTE_CORRECOES_RSLAQ_2026-05-03.md` documenta 8 bugs críticos já corrigidos (buffer stub, truncagem de stdout, warmup etc.). Suíte de testes saudável.

## 4. Arquitetura do loop de controle ATUAL (IPC semáforos + CSV)

Fluxo por período de decisão (hoje 10 ms; será 100 ms no cenário SQR):

```
ns-3 (C++, rslaq-sim.cc :: KpmAndControlCallback, linhas ~1172–1383)
  1. escreve KPMs em rslaq-kpms.txt        (header na linha ~1227)
  2. sem_post(g_semMetricsReady)           (≈ linha 1282)
  3. sem_timedwait(g_semControl, 50 ms)    (≈ linha 1296)
       → timeout: warning "no action from agent", MANTÉM config anterior
  4. lê rslaq_actions_for_ns3.csv (cols[1]=sliceId, cols[2]=dedicatedPRB, cols[5]=algorithm)
  5. renormaliza pesos para soma 1 e chama
     g_schedulerPtr->SetSliceConfiguration(p_final, algos)   (≈ linha 1364)

Python (ns-o-ran-gym, RslaqEnv.step)
  1. bloqueia em metricsReadySemaphore.acquire(timeout=10 s)
  2. faz parse de rslaq-kpms.txt (rslaq_kpis.parse_kpm_file)
  3. agente decide ação
  4. action_controller.create_control_action escreve rslaq_actions_for_ns3.csv
  5. controlSemaphore.release()   (semáforo POSIX nomeado /sem_control_<simUuid>)
```

**Formatos exatos (contrato vigente):**

- `rslaq-kpms.txt` (uma linha por UE por período):
  `timestamp,ueImsi,sliceId,dTxBytes,dRxBytes,plr,resourceSharePct,dLostPackets,throughputMbps,bufferBytes`
- `rslaq_actions_for_ns3.csv` (uma linha por slice):
  `timestamp,sliceId,dedicatedPRB,minPRB,maxPRB[,algorithm]` — ex.: `500,1,40.0,40.0,100.0,2`
  (`dedicatedPRB/minPRB/maxPRB` são **percentuais** 0–100; algorithm 0=RR(custo 1,0), 1=PF(2,0), 2=BCQI(2,0))

**Extensões do contrato para o cenário SQR (Fase 0/1 — a congelar antes de codar):**
- Ação: formato inalterado. O agente SQR decide inteiros `a_i ∈ {0..C}` (C = PRBs totais da célula) e converte `dedicatedPRB = 100·a_i/C`; a restrição `Σa_i ≤ C` é garantida pela projeção no lado Python (o C++ já renormaliza pesos).
- KPM: acrescentar por UE `windowThroughputMbps` (throughput da janela), `snrDb`, `queueBytes` — lowest-user-throughput, p5 e nº de UEs por slice derivam no Python.

## 5. Regras de ouro (obrigatórias para todo agente)

1. **Não modificar `contrib/nr`** (patches defensivos lá dentro; hoje rastreado pelo fork Jussi_SBRC). Exceções: adicionar `contrib/nori` novo na Fase 5 (não sobrescreve nada) e **novos patches defensivos** — somente com autorização explícita do orquestrador, backup em `MODA/docs/patches_nr/` e registro da divergência em `MODA/docs/sqr/` (precedente: 4º patch `nr-qos-rule-classifier.cc`, 12/10/2026).
2. **Código novo isolado**: C++ em `MODA/ns-3-dev/scratch/sqr/`; Python em `MODA/ns-o-ran-gym/src/nsoran/sqr/` + `examples/sqr_*.py`. **Não alterar** `scratch/rslaq`, `scratch/our_paper` nem o core do ns-o-ran-gym (`rslaq_env`, `rslaq_reward`, `rslaq_kpis`, `action_controller`) sem registrar o motivo.
3. **Rastreabilidade de premissas** (regra do GUIA): todo valor não publicado pelo artigo entra em `MODA/docs/sqr/assumptions_register.md` marcado `[ASSUMIDO]` com justificativa; `[ARTIGO]` exige citação de seção/tabela. **Nunca apresentar valor assumido como publicado.**
4. **`NS3_DIR`**: `run_all_scenarios.sh` aponta para caminho antigo (`/home/eliothluy/Documentos/artigo_jussi/ns-3-dev`) — sempre sobrescrever: `NS3_DIR=/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev`.
5. **Sequência de validação**: testes sintéticos → smoke test curto (`./ns3 run sqr-sim` com controlador dummy) → campanha multi-seed. Nunca rodar campanha antes dos testes passarem.
6. **Percentuais do artigo (60%/20%) NÃO são critérios de aceitação** — recalcular tudo no nosso ambiente (o próprio artigo é internamente inconsistente na Seção V vs Conclusão).
7. **γ alterna SOMENTE o termo de custo** na seleção de ação (`f_τ(z,a) − γ·a/C`); a penalização de falha crítica `P_penalty=2` permanece ativa nas duas variantes (γ=0 e γ=1).
8. Commits: perguntar antes de commitar/pushar; o repo MODA tem commit único ("Initial commit") — estabelecer histórico limpo por fase.

## 6. Comandos úteis

```bash
# Build do ns-3 (a partir de MODA/ns-3-dev) — SEMPRE via gate wrapper (Python 3.13):
bash MODA/ns-3-dev/scratch/sqr/ns3_gate.sh build
# ou, dentro de MODA/ns-3-dev: bash scratch/sqr/ns3_gate.sh build   (não use ./ns3 direto com o python3.14 do sistema)

# Rodar o cenário atual (referência de funcionamento)
bash MODA/ns-3-dev/scratch/sqr/ns3_gate.sh run rslaq-sim   # parâmetros: ver padrões em run_all_scenarios.sh

# Campanhas de baseline existentes (17 modos × 5 cenários de tráfego)
cd /home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev
NS3_DIR=$PWD ./run_all_scenarios.sh   # SEMPRE com override de NS3_DIR

# Testes Python do ns-o-ran-gym
cd /home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym && python -m pytest tests/ -x -q

# Treino online existente (DDQN/SAC, referência de loop fechado)
cd /home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym && ./run_training.sh

# (Fase 1, futuro) cenário SQR
./ns3 run sqr-sim -- --simTime=600 --seed=1 ...
```

## 7. Papéis dos agentes e orquestração

### 7.1 Configuração de modelos (definida pelo usuário — 11/10/2026)

| Função | Modelo | Responsabilidades |
|---|---|---|
| **Orquestrador** | `account:zai-individual-coding-plan/GLM-5.3` (sessão principal, nível max) | Planejamento e decomposição das fases, dispatch dos implementadores, revisão de código, integração, decisões metodológicas e científicas, controle dos gatilhos go/no-go |
| **Implementadores** (Agentes A e B abaixo) | `account:zai-individual-coding-plan/GLM-5.3-Flash` (nível max) | Execução: escrever código C++/Python, rodar builds/testes/simulações, gerar logs e figuras |

- **Como fixar o modelo dos implementadores**: o orquestrador deve despachar o trabalho de implementação via **workflows dinâmicos** (`CreateWorkflow`/`AmendWorkflow`) com `subagent_model = "account:zai-individual-coding-plan/GLM-5.3-Flash"`. O tool `Agent` direto **não permite fixar o modelo** do subagente (segue o roteamento padrão do harness) — usar apenas para pesquisa/leitura ou quando o modelo não importar.
- Implementadores **não** alteram premissas do `assumptions_register.md`, contrato de interface ou decisões metodológicas sem aprovação do orquestrador.
- Verificar disponibilidade dos modelos antes de cada campanha: tool `ListModels`.

### 7.2 Papéis dos implementadores

| Agente | Perfil | Fases | Entregas principais |
|---|---|---|---|
| **A** | Especialista ns-3/5G-LENA + engenheiro C++ sênior | 1, 5.2, 5.3 | `scratch/sqr/sqr-sim.cc` (cenário fiel: Poisson 2 UE/min, sessão exp. 60 s, 3 eMBB, 2 GHz, 30 dBm, período 100 ms, KPMs estendidos); port do NORI para ns-3.46/v5.2; adaptador RIC Control → `SetSliceConfiguration` |
| **B** | Pesquisador sênior RL/ML | 0, 2, 3, 4, 5.4 | assumptions register; `src/nsoran/sqr/` (kernel Matérn, regressor quantílico pinball, política hierárquica, projeção); adaptador KBRL; campanha + figuras; xApp SQR |

- **Interface A↔B = contrato KPM/ação** (§4). O Agente B desenvolve contra um **mock** do contrato enquanto a Fase 1 avança (paralelismo sem bloqueio).
- **Gatilhos go/no-go** (herdados do GUIA §13):
  - SQR passa nos testes sintéticos (Fase 2) antes de qualquer run ns-3.
  - Cenário demonstra aplicação **real** de cota de PRB no MAC (log `prb_scheduled` por slice ≈ cota) antes da campanha.
  - Campanha IPC completa (Fase 4) antes da Fase 5.4 (xApp E2).

## 8. Estado atual — checklist das fases

| Fase | Escopo | Status |
|---|---|---|
| 0 | Contrato de interface + `assumptions_register.md` + `source_traceability.md` | ⬜ Pendente |
| 1 | `scratch/sqr/sqr-sim.cc` (cenário fiel) + KPMs estendidos + smoke | ⬜ Pendente |
| 2 | `src/nsoran/sqr/` (agente SQR completo) + `tests/test_sqr_*` + `examples/sqr_run.py` | ⬜ Pendente |
| 3 | Adaptador KBRL + baselines estáticos | ⬜ Pendente |
| 4 | Campanha multi-seed + Figs. 2–4 equivalentes + sensibilidade | ⬜ Pendente |
| 5 | NORI: infra RIC → port do módulo → adaptador controle → SQR como xApp → IPC vs E2 | ⬜ Pendente |

Detalhes, critérios de aceite e estimativas de cada fase: [`PLANO_IMPLEMENTACAO_SQR_NORI.md`](PLANO_IMPLEMENTACAO_SQR_NORI.md).

## 9. Compatibilidade NORI (resumo da viabilidade)

| Componente | NORI exige | MODA tem | Distância |
|---|---|---|---|
| ns-3 | commit `ab4cce02` = **ns-3.42** (2024-05-29) | ns-3.46-dev | 4 releases (adaptação de API) |
| 5G-LENA | commit de **2025-02-20** (era v5.x, `NrPmSearchFast`) | v5.2-dev + 3 patches | **pequena** |
| Scheduler RL | `nr-rl-mac-scheduler-ofdma` (linhagem GSoC/ns3-gym) | linhagem `nr-mac-scheduler-ofdma-ai` **já presente** no contrib/nr v5.2 | muito próxima |
| Compilador | GCC 11 recomendado (só p/ e2sim/ASN.1) | build atual OK (GCC moderno) | e2sim compila separado |
| Infra externa | Near-RT RIC O-RAN SC Release I (k8s) + e2sim + libsctp | não existe | **maior custo operacional** |

**Veredito**: viável com port moderado (dias, não semanas). O módulo NORI é pequeno (12 pares de classes) e a terminação E2 (e2sim + ASN.1/SCTP) é independente do ns-3. Risco principal: infraestrutura do RIC — por isso NORI é fase final com fallback IPC garantido.

## 10. Referências

- Artigo SQR: DOI [10.1109/CCNC65079.2026.11366589](https://doi.org/10.1109/CCNC65079.2026.11366589)
- NORI: <https://github.com/lasseufpa/nori> · e2sim (fork): <https://github.com/lasseufpa/e2sim> · Paper NORI: "Enabling NS-3 Simulations Integrated with Latest Versions of O-RAN Software" (de Oliveira et al., LASSE/UFPA — Biblioteca SBRT)
- KBRL (baseline): <https://github.com/jjalcaraz-upct/network-slicing> (Alcaraz et al., IEEE TWC 2022)
- 5G-LENA: <https://cttc-lena.gitlab.io/nr/html/> · repo: <https://gitlab.com/cttc-lena/nr>
- ns-3: <https://www.nsnam.org/documentation/>
- ns-O-RAN / OpenRAN Gym: <https://openrangym.com> · ns3-gym: <https://github.com/tkn-tub/ns3-gym>
- ns-O-RAN-flexric (alternativa E2, não adotada): <https://github.com/Orange-OpenSource/ns-O-RAN-flexric>

---

*Última atualização: 2026-10-11 — criado a partir das auditorias dos 3 agentes de pesquisa (artigo SQR, ns-3-dev, ns-o-ran-gym) + análise de viabilidade NORI.*
