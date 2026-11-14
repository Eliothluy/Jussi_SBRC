# Etapa 2.4-II-d-v2 — Expansão de Suporte eMBB e Recalibração Final do Oracle 3-Way

## 1. Objetivo

Executar uma única recalibração final, curta e focalizada, após a verificação independente da Etapa 2.4-II-d, atacando somente os dois bloqueadores estruturais ainda abertos:

1. o suporte atual do oracle em `a_e ∈ [18,37]` exclui ações eMBB acima de 37 RBGs que já foram demonstradas executavelmente como capazes de recuperar SLAs;
2. a banda de incerteza eMBB em alta carga permanece excessivamente larga/próxima do limiar, inflando artificialmente a classe `MARGINAL`.

A II-d-v2 NÃO deve reabrir:

- a modelagem causal da URLLC;
- a classificação 3-way;
- os SLAs;
- o cenário físico;
- o scheduler;
- o TDD;
- o mecanismo de replay;
- a microcalibração mMTC já concluída;
- os thresholds do gate após início da execução.

---

## 2. Estado confirmado de entrada

A verificação independente estabeleceu:

```text
30/31 observações do agente confirmadas
1 erro de estampa temporal corrigido
```

Matriz final verificada nas 22 testemunhas:

```text
             all-OK   violou
FEASIBLE        5        0
MARGINAL        9        5
INFEASIBLE      0        3
```

Portanto:

```text
FIR = 0.000
FFR = 0.000
```

nas classes confidentes.

O detector 3-way mostrou comportamento adequado:

```text
FEASIBLE    -> alta confiança de solução
MARGINAL    -> região de incerteza
INFEASIBLE  -> alta confiança de falha
```

O STOP global permaneceu apenas por bloqueadores estruturais do suporte/banda eMBB.

---

## 3. Achados científicos que ficam congelados

### 3.1. Violações evitáveis

Existem 5 violações evitáveis confirmadas por replay.

Em vários desses casos, a ação:

```text
(48,5,0)
```

ou equivalente de alta alocação eMBB recuperou o SLA eMBB no mesmo estado.

Essas ações demonstram que:

```text
a_e > 37
```

é uma região relevante do espaço de decisão.

### 3.2. URLLC

Para:

```text
a_u >= 13
```

a cauda URLLC mostrou-se majoritariamente:

```text
scheduler/TDD-dominated
```

e não resource-dependent.

Não refazer G4.

### 3.3. mMTC

Manter o piso já medido na II-d.

Não reabrir a calibração, exceto se alguma ação candidata violar explicitamente o piso congelado.

---

## 4. Hipótese principal da v2

A principal hipótese é:

> O excesso de `MARGINAL` em alta carga decorre parcialmente da falta de suporte eMBB acima de 37 RBGs e da consequente extrapolação/incerteza do modelo próximo ao SLA.

Formalmente, ações com `a_e > 37` podem ampliar `|F_t|` em estados atualmente classificados como `MARGINAL`.

---

## 5. Escopo da nova calibração

Expandir o suporte eMBB com replays executáveis em:

```text
a_e ∈ {38,40,42,44,46,48}
```

ou conjunto equivalente de 5–6 níveis cobrindo:

```text
38 <= a_e <= 48
```

Sem exceder:

```text
C = 53
```

e respeitando todos os pisos congelados das outras slices.

---

## 6. Construção das ações

Toda nova ação deve satisfazer:

```text
a_e + a_u + a_m <= 53
a_u >= a_u_min
a_m >= a_m_min
```

Usar os pisos vigentes do register.

Nunca criar ação apenas para maximizar eMBB violando os pisos já validados.

---

## 7. Casos obrigatórios de calibração

Reutilizar prioritariamente os estados que já demonstraram alta sensibilidade ao eixo eMBB.

Incluir obrigatoriamente:

- os 5 casos de violação evitável confirmada;
- os 3 antigos D3 críticos;
- os 2 casos de inviabilidade corroborada;
- algumas janelas de alta carga classificadas `MARGINAL`.

---

## 8. Número de replays

Meta:

```text
10–15 replays
```

Preferir desenhos que reutilizem a mesma janela com múltiplos valores de `a_e`.

Não ampliar campanha sem necessidade.

---

## 9. Princípio de seleção

A seleção de janelas deve ser pré-registrada antes da execução.

Critérios aceitáveis:

- proximidade do SLA eMBB;
- classificação `MARGINAL`;
- antiga contraprova executável;
- antigo caso `INFEASIBLE`;
- região de baixa CQI/alta carga;
- ação eMBB observada próxima ao limite 37.

Não selecionar com base no resultado pós-replay.

---

## 10. Saída mínima por replay

Gerar:

```csv
replay_id,
lambda_total,
seed,
window_id,
original_state_class,
cqi_embb,
n_embb,
queue_embb,
a_embb,
a_urllc,
a_mmtc,
embb_thr_median,
embb_sla_ok,
urllc_p95,
urllc_ok,
mmtc_pdr,
mmtc_ok,
joint_ok,
snr_p05,
replay_valid
```

Arquivo:

```text
support_extension_replays.csv
```

---

## 11. Curva executável eMBB

Para cada janela com múltiplos replays, construir:

```text
a_e
versus
throughput_eMBB
```

Verificar:

- monotonicidade aproximada;
- saturação;
- ganho marginal por RBG;
- ponto em que o SLA de 3,5 Mbps é atingido;
- dispersão entre estados.

---

## 12. Nova função eMBB

Retreinar/refinar somente o eixo eMBB.

Usar:

```text
a_embb
CQI contínuo
N_embb
queue_embb
offered_load_embb
util_active_dl
previous_thr_embb
```

Se disponível, incluir:

```text
snr
prb_scheduled
prb_executed
```

---

## 13. Proibição de regressão metodológica

Não voltar a bins grosseiros que criem degrau artificial próximo ao SLA.

Preferir:

```text
quantile regression
```

ou modelo contínuo equivalente.

Produzir:

```text
p25
p50
p75
```

---

## 14. Regra de largura da banda

Definir:

```text
W_t = p75_t - p25_t
```

A largura não deve ser interpretada automaticamente como inviabilidade.

Registrar:

```text
band_width_embb
distance_to_sla
support_density
```

---

## 15. Distância ao suporte

Criar métrica de suporte local, por exemplo:

```text
nearest_action_distance
local_sample_count
```

ou outra medida explícita de densidade.

Objetivo:

> distinguir banda larga por incerteza real de banda larga por falta de amostras.

---

## 16. Regra para MARGINAL

Manter a semântica:

```text
MARGINAL = banda cruza o SLA
```

Mas registrar também a causa predominante, se possível:

```text
MARGINAL_NEAR_THRESHOLD
MARGINAL_LOW_SUPPORT
MARGINAL_HIGH_VARIANCE
```

---

## 17. Critério de banda eMBB

Para throughput:

```text
FEASIBLE   se p25 >= 3.5 Mbps
INFEASIBLE se p75 < 3.5 Mbps
MARGINAL   caso contrário
```

Essa regra não pode ser alterada depois dos replays.

---

## 18. Ordem obrigatória

```text
1. rodar replays de suporte
2. validar integridade
3. anexar dados
4. refit eMBB
5. recalcular bandas
6. reprocessar estados
7. executar gate
```

Não refitar parcialmente antes de concluir os replays selecionados.

---

## 19. Casos de regressão obrigatórios

### R1 — 5 evitáveis confirmados

Eles não podem ser classificados como:

```text
INFEASIBLE
```

Se permanecerem `MARGINAL`, justificar quantitativamente.

### R2 — 3 antigos D3

Os casos com contraprova all-OK não podem ser `INFEASIBLE`.

### R3 — 2 inviáveis corroborados

Eles devem permanecer:

```text
INFEASIBLE
```

ou, no máximo:

```text
MARGINAL
```

com justificativa de proximidade de fronteira.

### R4 — FEASIBLE antigos

Os 5 FEASIBLE confirmados não podem regredir para `INFEASIBLE`.

---

## 20. Gate final da v2

A II-d-v2 só pode dar PASS se:

### C1 — FIR

```text
FIR = 0
```

ou no máximo:

```text
FIR <= 0.10
```

nas testemunhas confidentes.

### C2 — FFR

```text
FFR = 0
```

ou:

```text
FFR <= 0.10
```

### C3 — suporte das evitáveis

As 5 violações evitáveis confirmadas devem possuir pelo menos uma witness dentro do suporte e compatível com `FEASIBLE` ou `MARGINAL`.

### C4 — antigos D3

Nenhuma contraprova executável all-OK pode permanecer `INFEASIBLE`.

### C5 — inviáveis corroborados

Os casos fortes de inviabilidade devem continuar coerentes com a fronteira.

### C6 — suporte

Nenhuma ação ótima relevante pode cair fora do suporte calibrado.

### C7 — MARGINAL

A classe `MARGINAL` deve continuar absorvendo incerteza sem gerar falsos confidentes.

---

## 21. Condição de STOP

STOP se surgir qualquer novo problema estrutural relevante, por exemplo:

- FIR/FFR pioram;
- expansão de suporte contradiz o modelo;
- witnesses necessárias continuam fora do suporte;
- banda continua excessivamente larga mesmo após aumento de densidade;
- casos antes estáveis mudam de classe de forma inexplicável;
- novo eixo não modelado passa a dominar.

---

## 22. Regra de encerramento

Esta é a última recalibração planejada da Etapa 2.4.

Se houver STOP novamente por um novo problema estrutural:

```text
não abrir automaticamente II-d-v3
```

Encerrar a caracterização com:

```text
FEASIBLE
MARGINAL
INFEASIBLE
```

e reportar a região `MARGINAL` explicitamente no artigo.

Nova rodada só pode ser aberta por decisão metodológica específica do orientador.

---

## 23. Reprocessamento após PASS

Somente após PASS:

reprocessar todas as janelas da Rodada 2 e congelar:

```text
P_F(lambda)
P_M(lambda)
P_I(lambda)
```

---

## 24. P_inf final

Definir:

```text
P_inf(lambda) = P_I(lambda)
```

somente após o PASS.

Manter:

```text
P_M(lambda)
```

como métrica explícita de incerteza/fronteira.

---

## 25. AVR final

Calcular:

```text
AVR(lambda) = P(V=1 | state=FEASIBLE)
```

e também:

```text
AVR_FM(lambda) = P(V=1 | state∈{FEASIBLE,MARGINAL})
```

Reportar separadamente.

---

## 26. |F_t| final

Calcular somente dentro do espaço de ações suportado:

```text
|F_t|
|M_t|
```

---

## 27. D_min final

Calcular:

```text
D_min_support
```

somente no simplex calibrado.

Não usar nenhuma ação fora do suporte.

---

## 28. Escolha de L_N

Escolher uma carga com:

```text
P_I ≈ 0
P_M baixo
P_F alto
```

Provável candidato:

```text
lambda = 2
```

mas congelar apenas após PASS.

---

## 29. Escolha de L_O

Escolher carga com:

```text
0 < P_I < 1
```

e quantidade relevante de estados `MARGINAL`.

Prováveis candidatos:

```text
lambda = 5
ou
lambda = 6
```

Não usar automaticamente lambda 7.

---

## 30. Decisão final SQR-JC

Após PASS, usar:

```text
AVR
P_I
P_M
D_min
```

para selecionar componentes.

### Se AVR domina

```text
Joint Coordination
```

### Se P_I é relevante

```text
Feasibility Awareness
Graceful Degradation
```

### Se P_M é relevante

```text
Uncertainty-aware coordination
```

---

## 31. Regra para runtime futuro

### FEASIBLE

```text
coordenar para satisfazer todos
```

### MARGINAL

```text
agir conservadoramente
não declarar inviabilidade
evitar agressividade
```

### INFEASIBLE

```text
graceful degradation
```

---

## 32. Artefatos obrigatórios

Criar:

```text
oracle_recalibrated_v2/
├── support_extension/
├── models/
├── predictions/
├── adjudication/
├── plots/
├── tables/
└── reports/
```

Arquivos mínimos:

```text
support_extension_replays.csv
embb_support_curve.csv
embb_model_card_v2.md
oracle_3way_windows_v2.csv
oracle_3way_by_lambda_v2.csv
confusion_matrix_3way_v2.csv
gate_verdict_II_d_v2.json
ln_lo_selection_v2.md
relatorio_oracle_2_4_II_d_v2.md
```

---

## 33. Figuras obrigatórias

1. throughput eMBB × a_e para os replays de suporte;
2. largura da banda eMBB × a_e;
3. erro near-threshold antes × depois;
4. confusion matrix 3-way final;
5. P_F/P_M/P_I × lambda;
6. |F_t|/|M_t| × lambda;
7. mapa de estados final.

---

## 34. Relatório final

Estrutura:

```text
1. objetivo
2. motivo da v2
3. verificação independente que motivou a correção
4. expansão do suporte
5. curva eMBB executável
6. refit eMBB
7. banda de incerteza
8. regressão dos casos críticos
9. confusion matrix
10. gate final
11. P_F/P_M/P_I
12. AVR
13. |F_t|
14. D_min
15. fronteira final
16. L_N
17. L_O
18. decisão SQR-JC
19. limitações
20. veredito
```

---

## 35. Linguagem obrigatória

Usar:

```text
violação evitável confirmada por replay
```

somente para casos já demonstrados executavelmente.

Usar:

```text
inviabilidade corroborada
```

para os estados suportados por replay/oracle.

Usar:

```text
MARGINAL
```

quando a banda cruza o SLA.

Nunca converter `MARGINAL` em `INFEASIBLE` apenas para fechar a fronteira.

---

## 36. Pergunta científica da v2

> Ao ampliar o suporte eMBB para a região onde as ações alternativas realmente recuperam SLA, o detector 3-way consegue preservar FIR/FFR próximos de zero e reduzir a incerteza artificial da fronteira em alta carga?

---

## 37. Comando resumido para o agente

```text
Executar a Etapa 2.4-II-d-v2 como ÚLTIMA recalibração planejada da Rodada 2.

Escopo estrito:

1. manter congelados URLLC, mMTC, SLAs, cenário, scheduler, TDD, replay e regra 3-way;
2. expandir o suporte eMBB acima de 37 RBGs com ~10–15 replays executáveis pré-registrados, cobrindo aproximadamente a_e={38,40,42,44,46,48};
3. respeitar os pisos congelados de URLLC e mMTC;
4. reutilizar obrigatoriamente os 5 casos evitáveis, os D3 críticos, os 2 inviáveis corroborados e janelas MARGINAL de alta carga;
5. construir curva throughput_eMBB × a_e;
6. refit somente o modelo/banda eMBB, usando p25/p50/p75 e suporte local explícito;
7. manter a regra:
   FEASIBLE se p25>=3.5,
   INFEASIBLE se p75<3.5,
   MARGINAL caso contrário;
8. repetir o gate final sem alterar thresholds após observar os resultados;
9. se PASS, congelar P_F/P_M/P_I, P_inf, AVR, |F_t|, D_min, L_N e L_O;
10. decidir então a arquitetura final do SQR-JC.

Se houver novo STOP estrutural, NÃO abrir automaticamente II-d-v3.
Encerrar a caracterização reportando explicitamente a classe MARGINAL e levar a decisão ao orientador.
```
