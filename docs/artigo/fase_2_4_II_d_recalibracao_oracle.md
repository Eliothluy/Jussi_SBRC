# Etapa 2.4-II-d — Recalibração do Oracle de Viabilidade com Banda de Incerteza e Suporte Executável

## 1. Objetivo

Recalibrar o oracle offline após o **STOP da Etapa 2.4-II-c**, corrigindo os erros sistemáticos identificados na validação executável e submetendo o novo detector de viabilidade ao **mesmo padrão de gate executável**, sem relaxar os critérios a posteriori.

A Etapa 2.4-II-d deve responder:

1. é possível reduzir os falsos `INFEASIBLE` observados em baixa carga?
2. é possível reduzir os falsos `FEASIBLE` observados em estados marginais?
3. o oracle consegue distinguir `FEASIBLE`, `MARGINAL/INCERTO` e `INFEASIBLE`?
4. o conjunto de ações do oracle pode ser restringido ao suporte observado sem perder a capacidade de detectar inviabilidade?
5. o modelo eMBB em CQI baixo pode ser recalibrado próximo ao SLA de 3,5 Mbps?
6. o gate URLLC pode representar dispersão de janela sem tratar a cota como causa quando o G4 mostrou scheduler/TDD dominante?
7. após a recalibração, P_inf, AVR, |F_t|, D_min, L_N e L_O podem ser congelados?

---

## 2. Motivação

A Etapa 2.4-II-c encerrou com:

```text
GATE = STOP
```

Principais causas:

- false-infeasible elevado;
- contraprovas executáveis em G3;
- falsos FEASIBLE no eixo URLLC;
- super-fechamento do modelo eMBB em CQI baixo;
- ações ótimas do oracle fora do suporte observado;
- D_min irreal quando o oracle escolhe a_u=0 ou a_m=0.

Achados que permanecem válidos:

- replay executável determinístico;
- 28/28 replays íntegros;
- 5 violações evitáveis confirmadas por replay;
- joint coordination é um problema real;
- cauda URLLC é scheduler/TDD-dominada para a_u >= 13;
- a_u=0 causa colapso físico de URLLC;
- a_m=0 pode derrubar o PDR do mMTC;
- o oracle binário atual não é confiável para congelar a fronteira.

---

## 3. Regra principal desta fase

O objetivo não é “fazer o oracle passar”.

O objetivo é:

```text
reduzir erro de classificação
+
representar incerteza explicitamente
+
respeitar suporte observado
+
preservar falsificabilidade
```

Se o novo detector continuar inconsistente com os replays:

```text
STOP novamente
```

---

## 4. Nova decisão em três faixas

Substituir a classificação binária direta:

```text
FEASIBLE
INFEASIBLE
```

por:

```text
FEASIBLE
MARGINAL
INFEASIBLE
```

A classe `MARGINAL` representa um estado em que o modelo não possui evidência suficiente para afirmar com confiança nem viabilidade nem inviabilidade.

---

## 5. Regra por faixa de incerteza

Usar bandas p25 / p50 / p75 por KPI.

Para KPI “maior é melhor”, como throughput e PDR:

- `FEASIBLE` se p25 >= SLA;
- `INFEASIBLE` se p75 < SLA;
- `MARGINAL` caso contrário.

Para KPI “menor é melhor”, como latência:

- `FEASIBLE` se p75 <= SLA;
- `INFEASIBLE` se p25 > SLA;
- `MARGINAL` caso contrário.

---

## 6. Classificação conjunta da ação

Para uma ação a:

### JOINT_FEASIBLE
Todos os KPIs críticos estão `FEASIBLE`.

### JOINT_INFEASIBLE
Pelo menos um KPI crítico está `INFEASIBLE`.

### JOINT_MARGINAL
A ação não é `JOINT_FEASIBLE` nem `JOINT_INFEASIBLE`.

Definir:

```text
F_t = ações JOINT_FEASIBLE
M_t = ações JOINT_MARGINAL
I_t = ações JOINT_INFEASIBLE
```

---

## 7. Classificação do estado

### FEASIBLE

Se:

```text
|F_t| > 0
```

### MARGINAL

Se:

```text
|F_t| = 0
|M_t| > 0
```

### INFEASIBLE

Se:

```text
|F_t| = 0
|M_t| = 0
```

Registrar:

```text
num_feasible_actions
num_marginal_actions
num_infeasible_actions
```

e depois:

```text
P_F(lambda)
P_M(lambda)
P_I(lambda)
```

com:

```text
P_F + P_M + P_I = 1
```

---

## 8. Refit do modelo eMBB

Problema identificado no II-c:

```text
thr_hat ≈ 3,42 Mbps
SLA = 3,5 Mbps
```

A margem de 0,08 Mbps ficou muito abaixo do erro near-threshold observado.

O novo modelo deve evitar bins grosseiros de CQI.

Preferir:

```text
CQI contínuo
N_embb
queue_embb
offered_load_embb
a_embb
util_active_dl
previous_thr_embb
```

Treinar estimativas:

```text
p25
p50
p75
```

ou modelo quantílico equivalente.

---

## 9. Gate de erro eMBB

Avaliar separadamente:

```text
overall MAE
near-threshold MAE
CQI-low MAE
classification error around 3.5 Mbps
```

Definir região near-threshold, por exemplo:

```text
|throughput - 3.5| <= 1 Mbps
```

Registrar a cobertura dos quantis.

---

## 10. Refit URLLC

Achado causal que deve ser preservado:

```text
para a_u >= 13, a cauda URLLC responde pouco à cota adicional
```

Portanto, acima desse piso, não modelar a latência como função monotônica simples de `a_urllc`.

Usar features de estado, preferencialmente:

```text
N_urllc
N_embb
queue_urllc
queue_embb
burst_indicator
dl_slots
TDD context/proxy
previous_latency
previous_deadline_miss
a_urllc
```

Modelar diretamente:

```text
p95_latency
```

ou:

```text
deadline_miss_ratio
```

por janela, com p25/p50/p75.

---

## 11. Gate de erro URLLC

Avaliar:

```text
MAE p95
erro near 10 ms
erro de classificação do SLA
coverage dos quantis
```

Near-threshold sugerido:

```text
|p95 - 10 ms| <= 2 ms
```

---

## 12. Piso de suporte URLLC

Congelar provisoriamente:

```text
a_u >= 13
```

Justificativa:

- G4 mostrou saturação de benefício acima de 13;
- G3 mostrou colapso em 0;
- ações entre 1 e 12 não estão suficientemente caracterizadas.

Se houver tempo, medir 1–12 como subexperimento separado. Não usar essa região no oracle final sem evidência.

---

## 13. Piso de suporte mMTC

Não permitir `a_m=0` no oracle final enquanto o piso mínimo não for calibrado.

Já observado:

```text
a_m=0 -> PDR pode cair para 0–0,67
a_m=3 -> PDR=1,0 nos casos observados
```

Executar microcalibração:

```text
a_m ∈ {0,1,2,3,4,5}
```

em algumas janelas com mMTC ativo, via replay determinístico.

Objetivo:

```text
encontrar a_m_min tal que PDR >= 95%
```

---

## 14. Novo espaço de ações

Após a calibração:

```text
a_u >= 13
a_m >= a_m_min
a_e >= 0
a_e + a_u + a_m <= 53
```

Proibir witnesses fora desse suporte.

---

## 15. Novo cálculo de D_min

Calcular apenas no espaço suportado:

```text
D_min_support = min_a sum_i w_i D_i(a)
```

para:

```text
a ∈ A_support
```

Gerar:

```text
D_min_p25
D_min_p50
D_min_p75
```

Se o intervalo cruzar zero, marcar:

```text
UNCERTAIN / MARGINAL
```

---

## 16. Treino e validação

Evitar leakage entre janelas correlacionadas.

Preferir:

```text
split por seed/run
```

ou:

```text
blocked temporal split
```

Se custo permitir, executar:

```text
leave-one-seed-out
```

Reportar:

```text
MAE
RMSE
classification accuracy
near-threshold error
quantile coverage
```

---

## 17. Reprocessamento completo

Após recalibração, reprocessar todas as janelas da Etapa 2.3.

Para cada janela gerar:

```text
state_class
num_feasible_actions
num_marginal_actions
num_infeasible_actions
best_feasible_action
best_marginal_action
best_degradation_action
D_min_support
```

Antes do novo replay gate, usar apenas nomes provisórios:

```text
P_F_candidate
P_M_candidate
P_I_candidate
```

Não produzir `P_inf_final`.

---

## 18. Seleção do novo replay gate

Criar grupos:

```text
D1 — antigos false-infeasible G1
D2 — antigos false-feasible G2
D3 — antigos G3 refutados
D4 — G3 corroborados
D5 — novas janelas MARGINAL
D6 — microcalibração mMTC
```

---

## 19. Reusar obrigatoriamente os casos difíceis

A nova validação deve incluir novamente:

- 7 false-infeasible do G1;
- 3 G3 com contraprova executável;
- 2 G3 corroborados;
- 3 falhas G2;
- janelas G4 como controle URLLC.

Não selecionar apenas casos novos favoráveis.

---

## 20. Expectativa por grupo

### D1
As 7 antigas janelas false-infeasible devem migrar para:

```text
FEASIBLE
```

ou:

```text
MARGINAL
```

### D2
Os antigos false-feasible devem ser absorvidos preferencialmente por `MARGINAL` ou classificados corretamente.

### D3
As três contraprovas G3 não podem continuar `INFEASIBLE` se a ação all-OK permanece válida e dentro do suporte.

### D4
Os dois casos corroborados devem permanecer `INFEASIBLE` ou `MARGINAL` fortemente próximo da borda.

### D5
Testar se a classe `MARGINAL` realmente concentra incerteza, e não apenas erro arbitrário.

### D6
Fechar `a_m_min`.

---

## 21. Confusion framework 3-way

Gerar matriz:

```text
FEASIBLE
MARGINAL
INFEASIBLE
```

Também manter duas visões conservadoras:

```text
FEASIBLE vs NOT-CONFIDENTLY-FEASIBLE
```

e:

```text
INFEASIBLE vs NOT-CONFIDENTLY-INFEASIBLE
```

Calcular:

```text
precision_feasible
recall_feasible
precision_infeasible
recall_infeasible
macro_f1
false_infeasible_rate
false_feasible_rate
marginal_rate
```

---

## 22. Gate de confiança II-d

O novo oracle só pode passar se:

### C1
FIR cair substancialmente.

Meta recomendada:

```text
FIR <= 0.25
```

### C2
FFR também cair.

Meta:

```text
FFR <= 0.25
```

### C3
Os 5 evitáveis confirmados não forem classificados como `INFEASIBLE`.

### C4
Os 2 G3 corroborados permanecerem coerentes com `INFEASIBLE` ou marginal extremo.

### C5
As 3 contraprovas G3 não forem classificadas como `INFEASIBLE`.

### C6
Nenhuma ação ótima usada para fechar o gate estiver fora do suporte.

### C7
Erros near-threshold forem absorvidos pela classe `MARGINAL`.

---

## 23. Regra anti-overfitting

Não alterar depois de ver o resultado final:

```text
thresholds
quantis
pisos
features principais
```

Qualquer alteração posterior deve abrir nova versão:

```text
II-d-v2
```

---

## 24. Após PASS

Somente após PASS recalcular e congelar:

```text
P_F(lambda)
P_M(lambda)
P_I(lambda)
```

A nova fronteira deve ser interpretada como:

```text
NORMAL:
P_F alto
P_M baixo
P_I ~ 0

MARGINAL:
P_M alto

OVERLOAD TRANSITÓRIO:
0 < P_I < 1

OVERLOAD SEVERO:
P_I -> 1
```

Somente então definir:

```text
P_inf(lambda) = P_I(lambda)
```

Mantendo `P_M(lambda)` como métrica principal da região de fronteira.

---

## 25. AVR revisado

Calcular:

```text
AVR = P(V=1 | state=FEASIBLE)
```

Opcionalmente:

```text
AVR_FM = P(V=1 | state∈{FEASIBLE,MARGINAL})
```

Reportar separadamente.

---

## 26. Escolha de L_N

Escolher regime com:

```text
P_I ~ 0
P_M baixo
P_F alto
```

Provavelmente lambda 2, mas não congelar antes do PASS.

---

## 27. Escolha de L_O

Escolher carga com:

```text
0 < P_I < 1
```

e presença relevante de:

```text
P_M
```

Preferir overload transitório, não colapso.

Lambda 5 ou 6 podem ser melhores candidatos que lambda 7.

---

## 28. Implicações para o SQR-JC

### Se AVR alto e P_I baixo

Priorizar:

```text
Joint Coordination
```

### Se P_I relevante

Adicionar:

```text
Feasibility Awareness
Graceful Degradation
```

### Se a classe MARGINAL for frequente

Adicionar explicitamente:

```text
Uncertainty-aware coordination
```

O controlador não deve tratar `MARGINAL` como `INFEASIBLE`.

---

## 29. Implicação URLLC

Como o G4 mostrou scheduler/TDD dominance:

```text
não tentar corrigir a cauda apenas aumentando cota
```

Acima de:

```text
a_u >= 13
```

tratar URLLC principalmente como gate de estado.

---

## 30. Implicação mMTC

Após obter `a_m_min`, preservar:

```text
a_m >= a_m_min
```

antes de redistribuir slack.

---

## 31. Implicação para joint coordination

### Estado FEASIBLE

```text
preservar pisos
+
distribuir slack
```

### Estado MARGINAL

```text
evitar decisões agressivas
+
preferir robustez
+
não declarar inviabilidade
```

### Estado INFEASIBLE

```text
graceful degradation
```

---

## 32. Artefatos esperados

```text
oracle_recalibrated/
├── models/
├── support/
├── predictions/
├── replay_v2/
├── adjudication/
├── plots/
├── tables/
└── reports/
```

Arquivos mínimos:

```text
embb_quantile_model_card.md
urllc_quantile_model_card.md
mmtc_floor_calibration.csv
oracle_3way_windows.csv
oracle_3way_by_lambda.csv
oracle_3way_replay_plan.csv
oracle_3way_replay_results.csv
confusion_matrix_3way.csv
gate_verdict_II_d.json
relatorio_oracle_2_4_II_d.md
```

---

## 33. Figuras obrigatórias

1. `P_F / P_M / P_I × lambda`
2. `|F_t| e |M_t| × carga`
3. erro eMBB near-threshold
4. erro URLLC near-threshold
5. PDR mMTC × a_m
6. confusion matrix 3-way
7. mapa `N_embb × N_urllc` com FEASIBLE/MARGINAL/INFEASIBLE

---

## 34. Relatório final II-d

Estrutura:

```text
1. objetivo
2. motivo do STOP anterior
3. alterações metodológicas
4. modelo eMBB recalibrado
5. modelo URLLC recalibrado
6. piso mMTC
7. espaço de ações suportado
8. oracle 3-way
9. replay gate
10. confusion matrix
11. P_F/P_M/P_I
12. AVR
13. D_min
14. fronteira
15. L_N
16. L_O
17. implicações para SQR-JC
18. limitações
19. veredito
```

---

## 35. Linguagem obrigatória

Usar:

```text
estado marginal
```

quando a banda de incerteza cruza o SLA.

Não forçar `FEASIBLE` ou `INFEASIBLE` nesses casos.

---

## 36. Critério de PASS

A Etapa 2.4-II-d só pode ser declarada PASS se:

- oracle 3-way passar no gate;
- falsos inviáveis forem reduzidos;
- falsos viáveis forem reduzidos;
- ações witness estiverem dentro do suporte;
- G2 evitáveis continuarem sustentados;
- G3 corroborados continuarem coerentes;
- nenhum erro sistemático equivalente ao II-c persistir.

---

## 37. Critério de STOP

STOP se:

- FIR continuar alto;
- FFR continuar alto;
- classe MARGINAL virar apenas um depósito de erro;
- modelos perto do SLA continuarem mal calibrados;
- suporte do oracle for insuficiente;
- mMTC não tiver piso confiável;
- replay contradizer sistematicamente o novo detector.

---

## 38. Próximo passo após PASS

Somente após PASS:

1. congelar P_F/P_M/P_I;
2. derivar P_inf=P_I;
3. congelar L_N;
4. congelar L_O;
5. definir arquitetura SQR-JC;
6. escrever especificação algorítmica;
7. implementar;
8. executar episódio transitório:

```text
L_N -> L_O -> L_N
```

9. comparar:

```text
SQR original
equal
prop
SQR-JC
oracle
```

---

## 39. Pergunta científica desta fase

> É possível construir um detector de viabilidade que represente explicitamente a incerteza próxima à fronteira e diferencie estados claramente viáveis, marginais e claramente inviáveis sem repetir os erros sistemáticos do oracle binário?

---

## 40. Comando resumido para o agente

```text
Executar a Etapa 2.4-II-d como recalibração do oracle após o STOP do II-c.

1. Implementar decisão 3-way: FEASIBLE / MARGINAL / INFEASIBLE.
2. Refit eMBB com CQI contínuo/mais fino e quantis p25/p50/p75.
3. Refit URLLC por janela com quantis e features de burst/estado; preservar o achado causal de que a_u >= 13 não controla a cauda.
4. Calibrar o piso mMTC com replay em a_m={0,1,2,3,4,5}.
5. Restringir o simplex às ações dentro do suporte: a_u >= 13 e a_m >= a_m_min.
6. Recalcular D_min somente no suporte.
7. Reprocessar todas as janelas da 2.3 com o oracle 3-way.
8. Construir plano de replay II-d reutilizando obrigatoriamente os casos críticos do II-c.
9. Rodar novo gate executável sem alterar thresholds após ver os resultados.
10. Se PASS, congelar P_F/P_M/P_I, P_inf, AVR, L_N e L_O.
11. Só depois decidir e especificar o SQR-JC.

Não implementar o SQR-JC antes do PASS do II-d.
```
