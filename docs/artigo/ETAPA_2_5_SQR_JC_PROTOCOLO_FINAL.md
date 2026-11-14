# ETAPA 2.5 — PROTOCOLO FINAL
## SQR-JC — Safe Quantile Regression with Feasibility-Aware Joint Coordination
### Protocolo de implementação, calibração, validação e campanha experimental

**Versão:** FINAL  
**Data de congelamento:** 2026-11-03  
**Status:** aprovado conceitualmente pelo orientador; pronto para execução após os gates descritos neste documento.  
**Objetivo:** implementar e avaliar a contribuição principal do novo artigo sem deixar decisões metodológicas críticas para serem tomadas implicitamente durante a execução.

---

# 0. Regra de autoridade deste documento

Este arquivo substitui as versões anteriores da Etapa 2.5.

O agente NÃO deve combinar instruções de versões antigas com este protocolo.

Se houver conflito entre:

```text
código atual
versão anterior do MD
anotações antigas
este documento
```

a ordem de prioridade é:

```text
1. este protocolo final
2. comportamento realmente implementado e auditado no código-base
3. artigo SQR original
4. resultados históricos da Rodada 2
```

Se o código contradizer o artigo ou este protocolo em um ponto que afete a validade experimental:

```text
STOP
registrar divergência
não corrigir silenciosamente
```

---

# 1. Pergunta científica principal

A contribuição do artigo é responder:

> **Dado o estado atual das slices e a capacidade disponível da célula, é possível satisfazer simultaneamente os SLAs? Se sim, qual alocação conjunta deve ser usada? Se a resposta for incerta, como agir com segurança? Se o estado for realmente inviável, como degradar o serviço de forma controlada?**

Princípio central:

> **Entender primeiro se, naquele estado, a violação é evitável e, a partir disso, decidir conjuntamente como distribuir os recursos.**

---

# 2. Identidade da proposta

A proposta deve permanecer uma extensão do SQR.

## Nome

```text
SQR-JC
```

Nome expandido:

> **Safe Quantile Regression with Feasibility-Aware Joint Coordination**

## Não chamar a proposta simplesmente de

```text
SQR
```

porque o SQR original será baseline.

Na escrita:

```text
Original SQR / SQR-HET baseline
Proposed SQR-JC
```

---

# 3. O que o SQR-JC herda do SQR

A proposta deve preservar o backbone de aprendizagem do SQR:

```text
Model-Based Reinforcement Learning
Online Quantile Regression
Kernelized Function Approximation
Matérn Kernel
Support-Vector / Landmark Memory
Online Pinball-Loss Update
Bounded Memory
Prediction Uncertainty
Risk-Aware Online Adaptation
Dynamic Safety Quantile
```

Esperado no artigo original:

```text
support-vector memory B = 2000
quantile range tau ∈ [0.01, 0.05]
learning rate eta = 0.1
epsilon = 0.1
kNN K = 10
P_penalty = 2
```

### Regra importante

Esses valores são **referência do artigo**, não devem ser assumidos como verdade do repositório.

Antes de implementar o SQR-JC, o agente deve auditar a implementação atual e produzir:

```text
sqr_original_audit.md
```

com:

```text
kernel
kernel parameters
feature normalization
tau initial value
tau min/max
tau adaptation rule
M adaptation rule
epsilon
eta
K
P_penalty
memory size
memory pruning
uncertainty equation
decision period
action unit
projection rule
```

A campanha só pode usar parâmetros congelados após essa auditoria.

---

# 4. O que muda no SQR-JC

O avanço está principalmente na política de decisão.

## SQR / SQR-HET

Conceitualmente:

```text
modelo da slice eMBB   -> a_e
modelo da slice URLLC  -> a_u
modelo da slice mMTC   -> a_m

                ↓

se a_e + a_u + a_m > C:
    projeção / correção
```

## SQR-JC

```text
estado conjunto
    ↓
ações conjuntas candidatas
    ↓
modelos kernelizados de todas as slices
    ↓
FEASIBLE / MARGINAL / INFEASIBLE
    ↓
seleção conjunta da ação
```

A ação é:

\[
\mathbf{a}_t = (a_e,a_u,a_m)
\]

e a capacidade entra na definição do espaço de decisão.

---

# 5. Formulação kernelizada

Para cada slice `i`, manter modelo kernelizado online:

\[
f^{(i)}_{\tau,t}(x)
=
\sum_{j=1}^{B}
\alpha^{(i)}_j k(x,x_j)
\]

A diferença é a entrada.

## SQR-HET baseline

\[
x^{(i)}_t =
[
z^{(i)}_t,\,
a^{(i)}_t/C
]
\]

## SQR-JC

\[
x_t =
[
z^{joint}_t,\,
a_e/C,\,
a_u/C,\,
a_m/C
]
\]

Os três modelos principais passam a avaliar o efeito da mesma ação conjunta:

\[
f_{eMBB}(z^{joint},a_e,a_u,a_m)
\]

\[
f_{URLLC}(z^{joint},a_e,a_u,a_m)
\]

\[
f_{mMTC}(z^{joint},a_e,a_u,a_m)
\]

---

# 6. Não substituir o learner

Na versão principal é proibido substituir o backbone por:

```text
DQN
PPO
SAC
MLP
Random Forest
XGBoost
regressão quantílica offline convencional
```

Esses métodos só podem entrar como baseline/ablação futura.

---

# 7. Representação comum de SLA

Para evitar inconsistência entre KPIs “maior é melhor” e “menor é melhor”, o target dos modelos deve ser uma **margem normalizada de SLA**.

Definir, para cada slice:

\[
m_i(y_i)
\]

de modo que:

```text
m_i > 0  -> SLA satisfeito
m_i = 0  -> fronteira
m_i < 0  -> SLA violado
```

## eMBB

Se o KPI for throughput:

\[
m_e =
\frac{T_e - SLA_e}{SLA_e}
\]

onde `T_e` é exatamente o KPI de throughput congelado no `sla_manifest`.

## URLLC

Se o KPI operacional for atraso:

\[
m_u =
\frac{D_{SLA}-D_u}{D_{SLA}}
\]

Se o código da Rodada 2 utilizar miss rate como critério primário:

\[
m_u =
\frac{MR_{SLA}-MR_u}{MR_{SLA}}
\]

Usar **somente uma definição primária**, conforme o código auditado.

## mMTC

\[
m_m =
\frac{PDR_m-PDR_{SLA}}{PDR_{SLA}}
\]

---

# 8. Hard gate — SLA manifest

Antes de rodar SQR/SQR-JC, criar:

```text
sla_manifest.yaml
```

contendo, para cada slice:

```text
KPI exato
unidade
agregação temporal
agregação por UE
limiar
direção do KPI
fórmula da margem normalizada
```

Esperado da Rodada 2:

```text
eMBB:
    throughput threshold ≈ 3.5 Mbps

URLLC:
    payload 32 B
    periodicidade 1 ms
    deadline experimental 10 ms app-to-app

mMTC:
    payload 200 B
    periodicidade 1000 ms
    PDR SLA = extrair do código congelado
```

### Proibido

Inventar ou alterar SLA durante a implementação.

---

# 9. Quantis: definição final

O protocolo utiliza **dois tipos de cabeça quantílica**.

## 9.1 Quantil de segurança herdado do SQR

Manter o mecanismo adaptativo do SQR:

\[
\tau^{safe}_i(t)
\]

com faixa esperada do artigo:

\[
[0.01,0.05]
\]

O código auditado é a fonte de verdade.

Esse modelo continua sendo o principal modelo de segurança.

---

## 9.2 Quantis auxiliares de viabilidade

Adicionar, para cada slice, dois modelos auxiliares:

```text
q25 = tau 0.25
q75 = tau 0.75
```

Eles devem usar:

```text
mesmo tipo de kernel Matérn
mesma representação de estado/ação do SQR-JC
mesmo mecanismo online de pinball loss
mesma política de memória
```

Esses quantis NÃO substituem o quantil seguro do SQR.

Eles servem apenas para:

```text
FEASIBLE
MARGINAL
INFEASIBLE
```

---

# 10. Memória dos quantis auxiliares

Implementação recomendada:

```text
mesmos support-vector locations
coeficientes separados por quantil
```

Por slice:

```text
alpha_safe
alpha_q25
alpha_q75
```

Objetivo:

```text
evitar três memórias espaciais completamente diferentes
reduzir overhead
manter suporte comparável
```

Se a arquitetura atual não permitir compartilhamento de support locations:

```text
documentar
medir overhead
não alterar silenciosamente
```


---

# 10.1. Quantile crossing

Como q25 e q75 são atualizados online separadamente, pode ocorrer:

\[
q25 > q75
\]

Isso é inconsistente.

Regra:

```text
1. registrar cada crossing;
2. para classificação somente, aplicar monotonic rearrangement:
   q25_used = min(q25_raw, q75_raw)
   q75_used = max(q25_raw, q75_raw)
3. manter os valores raw nos logs.
```

Métrica:

```text
quantile_crossing_rate
```

Gate:

Se:

```text
crossing rate > 5%
```

em CAL ou PILOT para qualquer slice:

```text
STOP
revisar treinamento/normalização/kernel
```

Não esconder crossing apenas reordenando os valores.

---

# 11. Estado conjunto

O `z_joint` deve conter apenas informação disponível antes da decisão.

Base:

```text
N_active_eMBB
N_active_URLLC
N_active_mMTC

queue/backlog_eMBB
queue/backlog_URLLC
queue/backlog_mMTC

previous_KPI_eMBB
previous_KPI_URLLC
previous_KPI_mMTC

offered_load_eMBB
offered_load_URLLC
offered_load_mMTC

CQI/SNR summary por slice

previous_a_e
previous_a_u
previous_a_m

recent_violation_eMBB
recent_violation_URLLC
recent_violation_mMTC
```

URLLC:

```text
recent miss rate
p95 latency
p99 latency
```

mMTC:

```text
active devices
recent PDR
pending/retransmission state
```

---

# 12. Proibição de leakage temporal

Nenhuma feature em `z_t` pode usar:

```text
KPI da janela t+1
resultado da ação ainda não executada
informação futura de chegada
resultado do oracle
classe de viabilidade calculada posteriormente por replay
```

Criar teste unitário:

```text
test_no_future_information()
```

---

# 13. Normalização das features

Kernel depende fortemente da escala.

Por isso a normalização é hard gate.

Primeiro:

```text
auditar normalização do SQR original
```

Features herdadas devem manter a mesma normalização, salvo justificativa.

Para novas features:

usar normalização física/semântica quando houver escala natural:

```text
allocation -> /C
KPI -> /SLA
delay -> /deadline
PDR -> natural [0,1]
utilization -> [0,1]
```

Para features sem limite físico conveniente, como contagens/filas:

```text
1. usar somente dados CAL;
2. calcular p1 e p99;
3. clip em [p1,p99];
4. mapear linearmente para [0,1];
5. congelar os valores.
```

Registrar tudo em:

```text
feature_scaler_manifest.yaml
```

Nunca recalcular scaler utilizando PILOT ou MAIN.

---

# 14. Kernel hyperparameters

Regra primária:

```text
usar exatamente o Matérn e os parâmetros do SQR auditado
```

Se a nova dimensão causar falha clara de calibração:

é permitida **uma única etapa de calibração pré-main**.

Grid permitido:

```text
original_lengthscale × {0.5, 1.0, 2.0}
```

Não mudar a família de kernel.

Escolha:

```text
menor pinball loss em validation calibration set
```

Após escolha:

```text
FREEZE
```

Nunca reajustar com resultados da campanha principal.

---

# 15. Suporte e incerteza

Preservar a métrica de incerteza do SQR, esperada como:

\[
\sigma_i(x)
=
1-\max_j k(x,x_j)
\]

ou exatamente a implementação auditada.

Ações fora de suporte não podem ser usadas para declarar inviabilidade.

---

# 16. Calibração de `sigma_support`

Usar somente a fase CAL.

Para cada slice:

1. obter distribuição de `sigma`;
2. testar thresholds nos percentis:

```text
50
60
70
80
90
95
```

3. escolher o maior threshold que mantenha calibração adequada de q25/q75.

Critério de calibração:

para o conjunto suportado,

```text
P(y <= q25) deve ser compatível com 0.25
P(y <= q75) deve ser compatível com 0.75
```

Critério formal:

Como as janelas temporais são autocorrelacionadas, NÃO usar intervalo binomial/Wilson tratando cada janela como independente.

Usar:

```text
moving/block bootstrap
block length = 5 s
1000 resamples
95% bootstrap CI
```

PASS se:

```text
tau alvo estiver dentro do 95% block-bootstrap CI da cobertura empírica
```

e:

```text
erro absoluto de cobertura <= 0.10
```

Além disso:

```text
supported coverage >= 70% dos pontos CAL
```

Se a duração de 5 s for incompatível com o logging real, converter 5 s para o número exato de decision windows e registrar no manifest.

Se nenhum threshold satisfizer:

```text
STOP
revisar features/kernel
não iniciar PILOT
```

---

# 17. Cenário físico principal

Referência:

```text
3GPP TR 38.901
UMa
```

Descrição correta:

> **single-cell UMa scenario based on the TR 38.901 implementation of the frozen 5G-LENA version**

Não afirmar:

```text
full 3GPP deployment
19-site layout
raio de 300 m definido pelo 3GPP
```

---

# 18. Versões do simulador

Antes de qualquer run registrar:

```text
ns3_version
5g_lena_version
git_commit_ns3
git_commit_nr
compiler
OS
channel helper configuration
channel model class
channel condition class
```

### Regra

Não atualizar ns-3/5G-LENA automaticamente.

Se a versão atual suporta 3GPP UMa/38.901:

```text
FREEZE
```

Se não suporta:

```text
STOP
propor branch de upgrade
rerodar todos os baselines após upgrade
```

---

# 19. TR 36.942

Usar somente como:

```text
referência histórica de KBRL/Safe RAN
```

Não usar como justificativa principal do novo cenário NR.

---

# 20. Configuração física inicial

Esperado:

```text
single cell
UMa
radius = 300 m
Tx power = 30 dBm
bandwidth = 10 MHz
scheduler = PF
RLC = UM
TDD = configuração congelada da Rodada 2
numerology = configuração congelada da Rodada 2
```

300 m significa:

> **calibrated operating point used to isolate resource competition from coverage failure.**

---

# 21. Hard gate — unidade de recurso

Antes de usar qualquer ação como:

```text
(37,13,3)
(48,5,0)
C=53
```

o agente deve provar o que a unidade representa.

Criar:

```text
resource_unit_manifest.md
```

respondendo:

```text
C=53 representa PRB?
RBG?
scheduler budget quantum?
número de grupos internos?
outro?
```

Também responder:

```text
um "allocation unit" corresponde a quantos PRBs físicos?
o scheduler pode deixar budget sem uso?
budget não utilizado pode ser emprestado?
a ação representa cap, quota, reserva ou uso real?
```

### Regra

Até isso estar resolvido:

```text
C=53 é apenas valor histórico do controlador
```

Não escrever “53 PRBs” ou “53 RBGs” no artigo.

---

# 22. Hard gate — semântica da ação

Definir:

```text
ACTION_SEMANTICS
```

como uma das opções:

```text
BUDGET_PARTITION
ALLOCATED_USAGE
OTHER
```

## Se BUDGET_PARTITION

Ação pode ser definida com:

\[
a_e+a_u+a_m=C
\]

e o espaço possui, para `C=53`:

```text
1485 combinações inteiras
```

## Se ALLOCATED_USAGE

Usar:

\[
a_e+a_u+a_m\le C
\]

Para `C=53`:

```text
27720 combinações
```

ou representar idle explicitamente.

A escolha depende do código.

Nunca escolher `sum=C` apenas para reduzir custo computacional.

---

# 23. Período de decisão

Manter, salvo auditoria em contrário:

```text
decision_period = 100 ms
```

Alinhar:

```text
ação k
KPI resultante
timestamp da janela
```

com a regra de replay validada anteriormente.

Criar teste:

```text
test_action_k_controls_expected_window()
```

---

# 24. Rádio saudável — gate físico

Antes da aprendizagem:

rodar low-load sanity.

Requisitos:

```text
lambda_total = 2
3 seeds
300 s
```

PASS se:

```text
RLF não for causa relevante
não houver outage persistente por cobertura
nenhuma slice apresentar KPI ~0 sistematicamente por PHY
SNR/CQI permanecerem em regime operacional
```

Registrar:

```text
RLF count
SNR p5
SNR median
CQI distribution
PHY error indicators
```

Se falhar:

```text
STOP
recalibrar apenas cenário físico
não mexer no algoritmo simultaneamente
```

---

# 25. Tráfego

Chegadas por slice:

```text
independent Poisson
```

Se `lambda_total` for usado:

\[
\lambda_e =
\lambda_u =
\lambda_m =
\lambda_{total}/3
\]

salvo se o código da Rodada 2 congelado indicar outra regra.

Sessões:

```text
Exponential(mean = 60 s)
```

Registrar unidade de lambda:

```text
UE arrivals per minute
```

---

# 26. Regimes de carga

Candidatos:

```text
L_N = lambda_total 2
L_O1 = lambda_total 5
L_O2 = lambda_total 6
```

`L_O` principal será escolhido apenas na fase PILOT.

Critério:

```text
presença de mistura de regimes
sem colapso físico
transição observável
```

A escolha não pode ser baseada em “onde SQR-JC ganha mais”.

---

# 27. Baseline principal justo: SQR-HET

O artigo SQR original usa três slices eMBB.

A campanha principal possui:

```text
eMBB
URLLC
mMTC
```

Logo, o baseline principal deve ser:

```text
SQR-HET
```

Definição:

> SQR original adaptado apenas para utilizar os KPIs/SLA heterogêneos e a margem normalizada comum, mantendo a arquitetura de decisão slice-wise seguida de projeção.

SQR-HET deve manter:

```text
per-slice state/action model
kernel quantile regression
dynamic tau
uncertainty
original fallback/exploration
original projection
```

Não possuir:

```text
joint action evaluation
feasibility-aware selection
conditional floors do SQR-JC
graceful degradation SQR-JC
```

Assim a diferença principal fica isolada.

---

# 28. Outros baselines

Obrigatórios:

```text
Equal
Prop
SQR-HET
SQR-JC
```

Opcional:

```text
KBRL
```

Reprodução homogênea do SQR-paper:

```text
separada da campanha principal
```

---

# 29. Cold start — desenho final

Não usar um dataset externo pré-treinado como condição principal.

A avaliação principal deve preservar o caráter online do SQR.

## Main runs

Durante:

```text
0–100 s
```

todos os controladores aprendentes usam o MESMO controlador de bootstrap:

```text
PROP
```

e observam exatamente os mesmos dados.

Os modelos podem ser atualizados online nesse período.

A partir de:

```text
t = 100 s
```

cada controlador passa a decidir.

Isso cria uma inicialização comum sem fornecer informação futura.

---

# 30. Métricas do bootstrap

Não descartar o bootstrap.

Reportar separadamente:

```text
0–100 s shared bootstrap
100–200 s normal learned control
200–400 s overload
400–600 s recovery
```

Comparação de política SQR-HET vs SQR-JC começa em:

```text
t=100 s
```

---

# 31. Cold-start ablation

Executar depois da campanha principal:

```text
SQR-HET from empty memory at t=0
SQR-JC from empty memory at t=0
```

Objetivo:

```text
medir custo da maior dimensionalidade
avaliar sample efficiency
```

Não usar essa ablação para alterar o método depois.

---

# 32. Fase CAL independente

Antes do PILOT:

```text
CAL_SEEDS = {1001,1002,1003}
lambda = {2,5,6}
duration = 300 s
```

Dados CAL podem ser usados somente para:

```text
feature scaling
kernel calibration autorizada
sigma_support
runtime search design
unit tests
```

Não usar CAL para reportar desempenho principal.

---

# 33. PILOT

```text
PILOT_SEEDS = {2001,2002,2003}
```

Rodar:

```text
SQR-HET
SQR-JC
```

em:

```text
lambda={2,5,6}
300 s
```

Objetivo:

```text
estabilidade
runtime
floor behavior
support coverage
MARGINAL rate
starvation
logging
```

Não usar PILOT como conjunto final de resultado.

---

# 34. MAIN

```text
MAIN_RUNS = 10
RNG runs = 3001..3010
```

Usar as mesmas RNG runs em todos os controladores.

Perfil:

```text
0–100 s   : lambda=2, shared PROP bootstrap
100–200 s : lambda=2, learned controller
200–400 s : lambda=L_O
400–600 s : lambda=2
```

Duração:

```text
600 s
```

---

# 35. Espaço de ações conjunto

Ações obedecem à semântica auditada.

Nenhuma ação pode:

```text
ser negativa
exceder C
violar piso ativo
usar unidade diferente da do scheduler
```

---

# 36. Busca conjunta — referência

Implementar um avaliador exhaustivo:

```text
joint_action_oracle_evaluator
```

para uso offline/calibração/replay.

Esse avaliador é a referência para medir erro da busca runtime.

---

# 37. Busca conjunta — gate de runtime

Primeiro tentar busca exaustiva vetorizada.

Benchmark em:

```text
100 estados CAL
cobrindo low/mid/high load
```

## PASS exaustivo

Se:

```text
p95 decision time <= 80 ms
```

usar busca exaustiva no runtime.

Objetivo:

```text
20% de margem em relação ao período de 100 ms
```

---

# 38. Coarse-to-fine autorizado

Se o exaustivo falhar no gate:

implementar busca determinística `coarse-to-fine`.

Parâmetros podem ser ajustados SOMENTE com CAL.

Ela deve ser comparada com o oracle exaustivo em:

```text
300 CAL states held out
```

PASS se:

```text
feasibility-class agreement >= 98%
median normalized objective regret <= 2%
p95 normalized objective regret <= 5%
p95 decision time <= 80 ms
```

Se falhar:

```text
STOP
não iniciar MAIN
```

---

# 39. Não declarar inviabilidade a partir de busca incompleta

Se a busca runtime não for exaustiva:

```text
"não encontrei ação viável"
```

não equivale automaticamente a:

```text
INFEASIBLE
```

Para emitir `INFEASIBLE` online, uma das condições deve valer:

```text
1. busca exaustiva foi executada
ou
2. mecanismo de busca tem certificado validado de cobertura
```

Caso contrário:

```text
state_class = MARGINAL_UNRESOLVED
```

e usar fallback conservador.

Na análise final, `MARGINAL_UNRESOLVED` pertence à família MARGINAL.

---

# 40. Pisos condicionados ao estado

Não codificar permanentemente:

```text
URLLC = 13
mMTC = 3
```

como pisos universais.

Os pisos devem surgir do estado e das previsões.

---

# 41. Regra principal dos pisos

Para cada slice `i`:

\[
floor_i(z)
\]

deve ser o menor nível de recurso sustentado por evidência kernelizada segura no estado atual.

Em termos operacionais:

1. procurar ações suportadas;
2. identificar menor `a_i` para o qual o modelo seguro da slice satisfaz a margem;
3. verificar que essa ação não exige comportamento claramente inviável das outras slices;
4. usar esse valor como piso candidato.

---

# 42. No-demand rule

Se:

```text
N_active_i == 0
```

e:

```text
queue_i == 0
no pending retransmission
```

então:

```text
floor_i = 0
```

---

# 43. Fallback de piso

Se não houver evidência suportada suficiente:

usar fallback conservador.

Se a auditoria confirmar mesma unidade e semântica da Rodada 2:

```text
URLLC fallback = 13
mMTC fallback = 3
```

Caso contrário:

recalibrar fallback com CAL.

Procedimento:

```text
selecionar menor alocação que mantém SLA em >=95% dos estados CAL suportados
```

e congelar antes do PILOT.

---

# 44. Valores 5 e 1

Valores históricos:

```text
URLLC = 5
mMTC = 1
```

podem ser usados como:

```text
diagnostic anchors
```

mas NÃO como regra global.

---

# 45. Margem de incerteza original do SQR

O SQR original adiciona margem de recurso após seleção por slice.

No SQR-JC:

```text
NÃO adicionar margem post-hoc depois da ação conjunta
```

porque isso recriaria o problema:

```text
joint action
→ post-hoc change
→ possível violation of capacity
```

O princípio de segurança por incerteza é preservado por:

```text
support filtering
conditional floors
MARGINAL state
uncertainty-aware tie breaking
conservative fallback
```

Essa é uma mudança intencional da política, não do learner.

---

# 46. Classificação de uma slice em uma ação

Usar os quantis auxiliares da margem normalizada.

Para uma slice `i`:

## FEASIBLE

\[
q25_i(m_i|z,a) \ge 0
\]

## INFEASIBLE

\[
q75_i(m_i|z,a) < 0
\]

## MARGINAL

caso contrário.

### Regra de suporte

Se q25/q75 estiverem fora de suporte:

```text
slice_action_class = MARGINAL
```

Nunca `INFEASIBLE`.

---

# 47. Classificação de ação conjunta

## JOINT_FEASIBLE

Todas as slices:

```text
FEASIBLE
```

## JOINT_MARGINAL

Nenhuma slice é confidentemente INFEASIBLE e pelo menos uma é MARGINAL.

## JOINT_INFEASIBLE

Pelo menos uma slice é INFEASIBLE.

---

# 48. Classificação do estado

## FEASIBLE

Se existe pelo menos uma:

```text
JOINT_FEASIBLE action
```

## MARGINAL

Se:

```text
não existe JOINT_FEASIBLE
```

mas existe:

```text
JOINT_MARGINAL
```

ou há ações fora de suporte.

## INFEASIBLE

Somente se:

```text
todas as ações do espaço validado
```

forem:

```text
JOINT_INFEASIBLE
```

e a busca possuir cobertura/certificação adequada.

---

# 49. Regra absoluta

```text
MARGINAL != INFEASIBLE
UNKNOWN != INFEASIBLE
OUT_OF_SUPPORT != INFEASIBLE
```

---

# 50. Seleção em FEASIBLE — sem pesos arbitrários

Usar decisão lexicográfica.

Entre `JOINT_FEASIBLE`:

```text
1. menor risco quantílico agregado
2. menor incerteza total
3. menor switching distance em relação a a(t-1)
4. menor recurso total, se a semântica permitir sum<C
5. deterministic tie-break
```

Não usar pesos contínuos ajustados no MAIN.

---

# 51. Seleção em MARGINAL

Usar decisão lexicográfica:

```text
1. maximizar número de slices não-INFEASIBLE
2. maximizar pior q25 normalized SLA margin
3. minimizar incerteza total
4. minimizar switching
5. deterministic tie-break
```

---

# 52. Seleção em INFEASIBLE

Graceful degradation.

Usar primeiro:

```text
minimize maximum normalized violation severity
```

Depois:

```text
minimize sum of normalized violation severities
```

Depois:

```text
minimize switching
```

Isso produz uma estratégia:

```text
minimax fairness
```

sem introduzir prioridade arbitrária.

---

# 53. Prioridade URLLC

Não usar peso extra de URLLC na configuração principal.

Uma configuração:

```text
URLLC-priority
```

pode ser executada somente como ablação adicional.

---

# 54. Update online

Após executar:

\[
a_t=(a_e,a_u,a_m)
\]

observar:

```text
KPI_e
KPI_u
KPI_m
```

converter para:

```text
m_e
m_u
m_m
```

e atualizar:

```text
safe quantile model
q25
q75
```

para cada slice.

Feature:

\[
x_t=[z^{joint}_t,a_e/C,a_u/C,a_m/C]
\]

---

# 55. Atualização do SQR-HET

SQR-HET recebe os mesmos outcomes.

Mas sua feature continua:

\[
x_i=[z_i,a_i/C]
\]

e a política continua slice-wise + projeção.

---

# 56. Logging obrigatório

Por decisão:

```csv
time,
rng_run,
controller,
phase,
lambda_total,

N_e,
N_u,
N_m,

queue_e,
queue_u,
queue_m,

prev_a_e,
prev_a_u,
prev_a_m,

floor_e,
floor_u,
floor_m,

state_class,
search_mode,
num_actions_total,
num_actions_evaluated,
num_joint_feasible,
num_joint_marginal,
num_joint_infeasible,

selected_a_e,
selected_a_u,
selected_a_m,

safe_q_e,
q25_e,
q75_e,
sigma_e,

safe_q_u,
q25_u,
q75_u,
sigma_u,

safe_q_m,
q25_m,
q75_m,
sigma_m,

support_e,
support_u,
support_m,

quantile_crossing_e,
quantile_crossing_u,
quantile_crossing_m,

decision_time_ms
```

Após janela:

```text
observed KPI
normalized margin
SLA status
```

---

# 57. Logs de kernel

Registrar:

```text
support vector count
insertions
evictions
kernel evaluations
prediction time
update time
out-of-support count
tau_safe(t)
M_i(t), se ainda mantido no modelo auditado
```

---

# 58. Métricas primárias

A campanha terá duas métricas principais complementares.

## P1 — Joint SLA Violation Rate

Em todas as janelas controladas:

```text
100–600 s
```

medir:

```text
proporção de janelas em que pelo menos uma slice viola SLA
```

---

## P2 — Avoidable Violation Fraction por replay independente

Em uma amostra pré-registrada de janelas que apresentaram violação:

\[
AVF =
\frac{
\text{violações com uma ação alternativa all-SLA-OK}
}{
\text{janelas violadas submetidas ao replay}
}
\]

Esperado:

```text
AVF(SQR-JC) < AVF(SQR-HET)
```

Essa é a métrica causal ligada diretamente à contribuição.

---

# 59. Por que não usar apenas “FEASIBLE segundo o próprio SQR-JC”

É proibido definir a métrica principal como:

```text
violations in states that SQR-JC itself called FEASIBLE
```

sem validação independente.

Isso seria circular.

As conclusões de evitabilidade devem ser confirmadas por:

```text
oracle/replay independente
```

---

# 60. Amostragem para replay

Usar seleção aleatória reproduzível.

Por controlador e RNG run:

selecionar até:

```text
5 violated windows por fase
```

nas fases:

```text
100–200 normal
200–400 overload
400–600 recovery
```

Se houver menos de 5:

```text
usar todas
```

Se um controlador não apresentar nenhuma violação no conjunto elegível:

```text
AVF = NA (denominador zero)
confirmed avoidable violations = 0
```

Não substituir `NA` por zero, porque “nenhuma violação ocorreu” é diferente de “todas as violações foram inevitáveis”.

RNG de amostragem:

```text
fixo e independente do simulador
```

Além disso, pode haver uma amostra extra direcionada para casos qualitativos, mas ela NÃO entra no cálculo do AVF principal.

---

# 61. Replay

Para cada janela selecionada:

```text
same state
same random stream
same arrivals
same channel
same history through t-1
change only action at t
```

Testar ação observada + conjunto de alternativas.

Se existir alternativa:

```text
all-SLA-OK
```

a violação é:

```text
CONFIRMED_AVOIDABLE
```

Se nenhuma alternativa testada recuperar:

```text
NOT_RECOVERED_IN_ACTION_SPACE
```

Não escrever “matematicamente inevitável” sem exaustão/certificação.

---

# 62. Métricas secundárias

eMBB:

```text
median throughput
p5 throughput
SLA violation rate
violation severity
```

URLLC:

```text
miss rate
p50 latency
p95 latency
p99 latency
SLA violation rate
```

mMTC:

```text
PDR
PDR violation
```

Sistema:

```text
utilization
idle resources
scheduled/budget ratio
switching rate
decision time
```

---

# 63. Severity

Usar a margem normalizada.

Se:

\[
m_i<0
\]

então:

\[
severity_i=-m_i
\]

Caso contrário:

```text
severity = 0
```

Isso mantém comparabilidade entre slices.

---

# 64. Recovery time

Definir:

> tempo entre `t=400 s` e o primeiro instante em que todas as slices permanecem em SLA por uma janela sustentada predefinida.

Janela sustentada:

```text
5 s
```

Congelar antes do MAIN.

---

# 65. Starvation

Registrar:

```text
allocation == 0
allocation near floor
consecutive low-allocation windows
```

Reportar:

```text
max consecutive starvation duration
```

por slice.

---

# 66. Runtime overhead

Reportar:

```text
decision_time median
p95
p99
kernel prediction time
online update time
num evaluated actions
```

---

# 67. Quantile calibration metrics

Para q25/q75:

```text
empirical coverage
coverage error
pinball loss
```

Para safe tau:

```text
empirical underprediction rate
tau(t)
critical-failure rate
```

---

# 68. Ablations obrigatórias

Depois de congelar e executar a proposta completa.

## A0

```text
SQR-HET
```

## A1

```text
SQR + Joint Action Evaluation
sem conditional floors
sem feasibility-aware runtime
```

## A2

```text
Joint Coordination + Conditional Floors
sem graceful degradation específico
```

## A3

```text
SQR-JC completo
```

Objetivo:

```text
isolar contribuição de cada componente
```

---

# 69. Ablation da representação conjunta

Comparar:

```text
per-slice action feature
```

versus:

```text
joint action feature
```

sob mesma política conjunta, se computacionalmente possível.

---

# 70. Ablation MARGINAL

Comparar:

```text
3-way:
FEASIBLE / MARGINAL / INFEASIBLE
```

com:

```text
binary:
MARGINAL -> INFEASIBLE
```

Somente como ablação.

---

# 71. Estatística

MAIN:

```text
10 paired RNG runs
```

Reportar:

```text
mean
median
std
95% CI
```

Comparação pareada:

```text
same RNG run between controllers
```

Se adequado:

```text
paired t-test
```

Se não-normal:

```text
Wilcoxon signed-rank
```

Sempre reportar:

```text
effect size
```

Não usar p-value sozinho.

---

# 72. Múltiplas métricas

Não realizar dezenas de testes de hipótese sem controle.

Definir como primárias:

```text
P1 Joint SLA Violation Rate
P2 Avoidable Violation Fraction
```

Outras métricas:

```text
secondary/explanatory
```

---

# 73. Integridade de versões

Cada run deve gerar:

```json
{
  "protocol": "SQRJC-ETAPA-2.5-FINAL",
  "controller": "",
  "rng_run": 0,
  "git_commit_project": "",
  "git_commit_ns3": "",
  "git_commit_5glena": "",
  "ns3_version": "",
  "5glena_version": "",
  "channel_model": "",
  "scenario": "UMa",
  "resource_unit": "",
  "action_semantics": "",
  "C": null,
  "decision_period_ms": 100,
  "lambda_profile": "",
  "duration_s": 600
}
```

---

# 74. Diretórios

```text
results/sqr_jc_final/
├── 00_audit/
├── 01_resource_unit/
├── 02_radio_sanity/
├── 03_calibration/
├── 04_baseline_validation/
├── 05_pilot/
├── 06_main/
├── 07_ablations/
├── 08_replays/
├── 09_statistics/
├── plots/
├── tables/
└── reports/
```

Código:

```text
src/nsoran/sqr_jc/
├── kernel_models.py
├── sla_margin.py
├── joint_state.py
├── feature_scaler.py
├── joint_action_space.py
├── action_search.py
├── feasibility.py
├── conditional_floors.py
├── joint_controller.py
├── online_update.py
├── logging.py
└── tests/
```

---

# 75. Testes unitários obrigatórios

```text
test_resource_unit_manifest_complete
test_action_semantics
test_capacity_constraint
test_nonnegative_action
test_decision_window_alignment

test_sla_margin_embb
test_sla_margin_urllc
test_sla_margin_mmtc

test_kernel_prediction
test_kernel_support_insertion
test_memory_budget
test_memory_pruning
test_quantile_update
test_q25_q75_ordering
test_quantile_crossing_logging
test_monotonic_rearrangement_for_classification
test_uncertainty
test_support_threshold

test_no_future_information
test_joint_feature_vector
test_normalization

test_no_demand_floor_zero
test_conservative_floor_fallback

test_joint_feasible
test_joint_marginal
test_joint_infeasible
test_unknown_not_infeasible
test_marginal_not_infeasible

test_deterministic_tie_break
test_graceful_degradation

test_shared_bootstrap_determinism
test_replay_state_identity
```

---

# 76. Regression tests da Rodada 2

Se a unidade e semântica continuarem compatíveis, usar como testes qualitativos:

```text
casos recuperados por (48,5,0)
casos não recuperados por (37,13,3)
URLLC-sensitive windows
mMTC zero-allocation failure
robust high-load failures
```

Não exigir reprodução numérica idêntica após mudança do cenário físico.

---

# 77. Gates de execução

## G0 — SQR audit

PASS somente com:

```text
sqr_original_audit.md completo
```

## G1 — Resource semantics

PASS somente com:

```text
resource_unit_manifest.md
ACTION_SEMANTICS definido
```

## G2 — SLA semantics

PASS somente com:

```text
sla_manifest.yaml
```

## G3 — Radio sanity

PASS com rádio saudável.

## G4 — Kernel calibration

PASS com q25/q75 adequadamente calibrados e suporte >=70%.

## G5 — Search runtime

PASS com:

```text
p95 <=80 ms
```

e qualidade mínima, se aproximada.

## G6 — Baseline validity

SQR-HET deve rodar sem erro e manter comportamento coerente.

## G7 — PILOT

PASS se:

```text
sem crash
sem NaN
sem capacity violation
sem systematic starvation
logs completos
pisos não degenerados
classificação não totalmente MARGINAL
```

Somente depois:

```text
MAIN
```

---

# 78. Critério adicional do PILOT

Se mais de:

```text
80% das decisões
```

forem:

```text
MARGINAL / UNKNOWN
```

em todas as três cargas:

```text
STOP
```

Isso indica que o classificador não está informativo o suficiente.

Não mascarar o problema com thresholds escolhidos depois.

---

# 79. Escolha de L_O

Após PILOT escolher entre:

```text
5
6
```

Critérios, nesta ordem:

```text
1. rádio continua saudável
2. existem FEASIBLE/MARGINAL/INFEASIBLE ou transições úteis
3. não há colapso permanente
4. overload é suficientemente forte para gerar competição
```

Proibido:

```text
escolher L_O porque maximiza vantagem do SQR-JC
```

Registrar decisão em:

```text
load_regime_decision.md
```

antes do MAIN.

---

# 80. O que o agente pode ajustar

Antes do MAIN, somente dentro das regras deste protocolo:

```text
kernel lengthscale no grid autorizado
sigma_support
runtime search parameters
L_O entre 5 e 6
fallback floor se unidade histórica não for compatível
```

Todos usando CAL/PILOT, nunca MAIN.

---

# 81. O que o agente NÃO pode ajustar

Depois do início do MAIN:

```text
SLA
traffic
lambda
radius
Tx power
bandwidth
scheduler
TDD
kernel
tau rules
sigma_support
floors
search parameters
objectives
state features
normalization
bootstrap duration
```

Qualquer necessidade de alteração:

```text
invalidate MAIN
abrir nova versão de protocolo
```

---

# 82. Bug fixing

Bug de implementação:

```text
pode e deve ser corrigido
```

Mas:

1. registrar bug;
2. registrar commit;
3. dizer quais runs são afetados;
4. invalidar e rerodar runs afetados;
5. aplicar correção de forma simétrica aos controladores quando pertinente.

---

# 83. First deliverable do agente

Antes do PILOT entregar:

```text
docs/artigo/sqr_jc_preflight_report.md
```

com:

1. SQR audit;
2. ns-3/5G-LENA versions;
3. TR 38.901 UMa mapping;
4. resource unit;
5. action semantics;
6. SLA manifest;
7. joint feature vector;
8. normalization;
9. kernel parameters;
10. q25/q75 calibration;
11. support threshold;
12. action-space size;
13. exhaustive-search benchmark;
14. chosen runtime search;
15. physical sanity;
16. known blockers.

---

# 84. Segundo deliverable

Depois do PILOT:

```text
docs/artigo/sqr_jc_pilot_report.md
```

com:

```text
lambda 2/5/6
3 seeds
SQR-HET vs SQR-JC
runtime
support
class distribution
floors
SLA metrics
starvation
recommendation L_O
PASS/STOP
```

---

# 85. Terceiro deliverable

Depois do MAIN:

```text
docs/artigo/sqr_jc_main_report.md
```

com:

```text
10 paired runs
primary metrics
secondary metrics
statistics
phase analysis
replay sample
runtime
kernel calibration
failure cases
```

---

# 86. Figura central esperada

Para uma seed representativa escolhida por regra pré-definida, não por aparência:

```text
lambda(t)
state_class(t)
a_e(t), a_u(t), a_m(t)
SLA margins
floors
```

Comparar:

```text
SQR-HET
SQR-JC
```

---

# 87. Regra para seed representativa

Escolher a seed cuja métrica P1 esteja:

```text
mais próxima da mediana das 10 runs
```

Não escolher a “melhor figura”.

---

# 88. Tabela principal esperada

```text
Controller
Joint SLA Violation Rate
Avoidable Violation Fraction
eMBB violation
URLLC violation/miss
mMTC violation
Violation severity
Recovery time
Decision time p95
Out-of-support rate
```

---

# 89. Mensagem científica que o artigo deve testar

> **Safety-aware RAN slicing should not rely only on independent per-slice resource requests followed by capacity correction. By preserving SQR's online kernelized quantile-learning backbone while evaluating the joint resource-allocation vector under explicit feasibility uncertainty, the controller can distinguish avoidable allocation failures from overload states and adapt slice protection accordingly.**

---

# 90. Contribuições esperadas — ainda como hipóteses

Não escrever como resultado antes da campanha.

Contribuição metodológica proposta:

1. joint action representation no SQR;
2. state-conditioned protection floors;
3. three-way feasibility awareness;
4. graceful degradation para estados confirmadamente inviáveis;
5. validação causal por deterministic replay.

---

# 91. Distinção fundamental

O SQR-JC NÃO pretende:

```text
prever o futuro perfeitamente
provar matematicamente a viabilidade contínua de todo o sistema
eliminar todas as violações
```

Pretende:

```text
reduzir decisões evitavelmente ruins
usar incerteza explicitamente
não confundir desconhecido com impossível
degradar de forma controlada quando não há solução suportada
```

---

# 92. Interpretação correta de INFEASIBLE

No artigo usar:

> **infeasible within the validated discrete action space and model support**

quando a classificação vier do oracle/discreto.

Evitar afirmação:

> **physically impossible under all conceivable policies**

sem prova correspondente.

---

# 93. Relação com a Rodada 2

Rodada 2 estabeleceu:

```text
violação existe com rádio saudável
carga aumenta probabilidade de estado crítico
parte das violações é recuperável por ação alternativa
pisos fixos podem bloquear uma recuperação
há região MARGINAL
há estados robustamente difíceis/inviáveis
```

Etapa 2.5 transforma esses achados em controlador.

---

# 94. Relação com o artigo SQR

O SQR original aprende:

```text
KPI risk per slice
```

e tenta selecionar ações seguras.

SQR-JC mantém esse learner e muda:

```text
from independent safe proposals + projection
to joint feasibility-aware allocation
```

Essa frase deve permanecer central na metodologia.

---

# 95. Não reabrir problemas já encerrados

Não usar a Etapa 2.5 para repetir todo o sweep da Rodada 2.

Rodar apenas o necessário para:

```text
sanity
calibration
baseline
pilot
main
ablation
replay
```

---

# 96. Checklist para autorizar PILOT

```text
[ ] SQR audit concluído
[ ] Resource unit definido
[ ] Action semantics definido
[ ] SLA manifest congelado
[ ] Feature scaler congelado
[ ] Kernel congelado
[ ] q25/q75 calibrados
[ ] sigma_support congelado
[ ] rádio saudável
[ ] decision period validado
[ ] action search passou runtime gate
[ ] SQR-HET validado
[ ] logs validados
[ ] unit tests PASS
```

Se qualquer item falhar:

```text
NÃO RODAR PILOT
```

---

# 97. Checklist para autorizar MAIN

```text
[ ] PILOT completo
[ ] L_O escolhido sem olhar ganho relativo
[ ] nenhum systematic starvation
[ ] MARGINAL/UNKNOWN não domina >80%
[ ] runtime p95 <=80 ms
[ ] todos hyperparameters congelados
[ ] MAIN seeds congeladas
[ ] replay sampling rule congelada
[ ] protocolo commitado
```

---

# 98. Comando final ao agente

```text
Execute a Etapa 2.5 conforme este protocolo, sem usar decisões metodológicas
de versões anteriores.

Primeiro faça somente o PRE-FLIGHT:

1. audite o SQR original;
2. identifique a unidade real de C=53 e a semântica da ação;
3. gere o SLA manifest;
4. confirme a versão ns-3/5G-LENA e o cenário UMa/38.901;
5. valide o rádio saudável;
6. implemente a representação de margem de SLA;
7. implemente os modelos kernelizados joint-action;
8. adicione q25/q75 como heads auxiliares, sem substituir tau_safe;
9. calibre suporte apenas com CAL;
10. implemente e benchmarke a busca conjunta;
11. implemente SQR-HET;
12. rode todos os unit/regression tests.

ENTREGA OBRIGATÓRIA antes de qualquer PILOT:

docs/artigo/sqr_jc_preflight_report.md

Não iniciar PILOT se G0–G6 não estiverem PASS.

Após aprovação automática dos gates objetivos do protocolo:

13. rodar PILOT com runs 2001–2003 em lambda {2,5,6};
14. gerar sqr_jc_pilot_report.md;
15. escolher L_O apenas pelos critérios de regime;
16. congelar todos os parâmetros;
17. executar MAIN com runs 3001–3010;
18. executar replay independente;
19. executar ablations;
20. gerar sqr_jc_main_report.md.
```

---

# 99. Referências metodológicas principais

1. **3GPP TR 38.901**, *Study on channel model for frequencies from 0.5 to 100 GHz*. Usar UMa conforme implementação da versão congelada do 5G-LENA.
2. **Tuerxun & Nakao**, *Safe RAN Slicing in O-RAN: Minimizing SLA Violations via Model-Based Reinforcement Learning*, IEEE CCNC 2026.
3. **Alcaraz et al.**, *Model-Based Reinforcement Learning With Kernels for Resource Allocation in RAN Slices*, IEEE Transactions on Wireless Communications.
4. **5G-LENA / ns-3 documentation and release notes** correspondentes à versão realmente usada.

TR 36.942:

```text
historical context only
```

---

# 100. Critério de encerramento científico

A Etapa 2.5 só termina quando existir evidência suficiente para responder:

> **Mantendo o backbone MBRL e a regressão quantílica online kernelizada do SQR, a coordenação conjunta consciente da viabilidade reduz violações evitáveis e adapta melhor a proteção das slices do que a seleção independente seguida de projeção, sem introduzir degradação sistemática em URLLC/mMTC e com custo computacional compatível com o período de decisão?**

Até lá:

```text
SQR-JC = proposta
```

Depois da campanha:

```text
resultados determinam se as hipóteses são aceitas, rejeitadas ou qualificadas
```
