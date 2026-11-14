# Próximo Passo — Etapa 2.4-II-c: Adjudicação Executável do Oracle e Congelamento da Fronteira

## 1. Objetivo

Concluir a validação executável do oracle da Etapa 2.4, usando os 28 replays já lançados, e transformar os resultados em uma decisão metodológica final sobre:

1. validade do oracle offline;
2. existência de violações evitáveis;
3. existência de estados conjuntamente inviáveis;
4. mecanismo da cauda URLLC;
5. fronteira de viabilidade;
6. escolha de L_N e L_O;
7. arquitetura final do SQR-JC.

A Etapa 2.4-II-c **não deve implementar o SQR-JC ainda**. O objetivo é fechar a evidência que decidirá qual versão do SQR-JC realmente se justifica.

---

## 2. Estado de entrada

A Etapa 2.4-II-b já forneceu:

- replay controller determinístico;
- mapeamento ação ↔ janela validado;
- reprodução bit-a-bit do run original até a janela alvo;
- override restrito à janela de interesse;
- retorno à ação-base após a janela;
- capacidade C=53 explicitamente congelada;
- equal-split reproduzido como (18,17,18);
- 242 testes passando;
- plano de 28 replays dividido em quatro grupos.

Grupos:

```text
G1 — false-infeasible / baixa carga
G2 — candidatos a violações evitáveis
G3 — candidatos a inviabilidade conjunta
G4 — causalidade URLLC por variação de a_u
```

---

## 3. Regra principal

Não fechar a fronteira somente com o oracle offline.

A fronteira só pode ser congelada após:

```text
oracle offline
      +
replay executável
      +
adjudicação dos conflitos
```

---

## 4. Concluir os 28 replays

Antes de qualquer análise:

- verificar 28/28 completos;
- verificar `.done`;
- validar integridade dos CSVs;
- verificar timestamps monotônicos;
- verificar ação-base antes/depois;
- verificar override somente na janela alvo;
- garantir C=53;
- registrar retry/falha.

Se algum replay estiver corrompido:

```text
refazer somente o replay afetado
```

Não aceitar run parcial.

---

## 5. Saída mínima por replay

Gerar `oracle_replay_results.csv` com:

```csv
group,
replay_id,
lambda_total,
seed,
arm_original,
window_id,
time_start_ms,
time_end_ms,
base_action_embb,
base_action_urllc,
base_action_mmtc,
override_embb,
override_urllc,
override_mmtc,
oracle_predicted_feasible,
oracle_predicted_infeasible,
oracle_num_feasible_actions,
observed_embb_ok,
observed_urllc_ok,
observed_mmtc_ok,
observed_joint_ok,
embb_throughput,
urllc_deadline_miss,
urllc_latency_p95,
mmtc_pdr,
snr_p05,
rlf_count,
replay_valid
```

---

## 6. G1 — adjudicação de false-infeasible

Objetivo: testar se o oracle classifica como inviáveis estados que, na prática, ainda admitem uma ação satisfatória.

As janelas G1 usam testemunha reconstruída pela porta p25, e não pela porta mediana principal. Portanto, classificar G1 como:

```text
VALIDAÇÃO DE FALSE-INFEASIBLE SOB CRITÉRIO AUXILIAR
```

e não como validação direta da porta principal.

Se:

\[
oracle = infeasible
\]

e:

\[
replay = all\_SLAs\_OK
\]

então:

\[
FALSE\text{-}INFEASIBLE
\]

Calcular:

\[
FIR =
\frac{
N(oracle\ infeasible, replay\ feasible)
}{
N(oracle\ infeasible\ testado)
}
\]

Registrar:

```text
false_infeasible_count
false_infeasible_rate
```

### Gate crítico

False-infeasible é o erro mais perigoso para a narrativa.

Não congelar a fronteira se o oracle:

- classificar como inviáveis muitas janelas que o replay salva;
- errar sistematicamente perto dos SLAs;
- depender excessivamente da escolha da porta.

Se FIR for relevante:

```text
STOP
```

e revisar modelo, limiar, porta, features e extrapolação.

---

## 7. G2 — adjudicação de violações evitáveis

Objetivo: confirmar causalmente que havia estados nos quais o controlador original violou SLA, mas uma ação alternativa consegue satisfazer todos.

Para cada janela G2:

```text
controlador original violou
oracle declarou viável
ação alternativa foi executada
```

Se o replay satisfizer todos os SLAs:

\[
SLA_{eMBB}=OK
\]

\[
SLA_{URLLC}=OK
\]

\[
SLA_{mMTC}=OK
\]

então classificar como:

```text
AVOIDABLE CONFIRMED
```

Calcular:

\[
ACR =
\frac{
N(G2\ confirmados)
}{
N(G2\ executados)
}
\]

Registrar por caso:

```text
ação original
ação alternativa
delta throughput eMBB
delta deadline-miss URLLC
delta PDR mMTC
delta utilização
```

A evidência causal desejada é:

```text
mesmo estado
+
mesma seed
+
mesmos arrivals
+
mesmo canal
+
outra ação
=
SLA recuperado
```

---

## 8. G3 — adjudicação de inviabilidade conjunta

Objetivo: testar estados que o oracle classificou como candidatos fortes a:

\[
\mathcal F_t=\emptyset.
\]

O replay não deve ser descrito como prova matemática de que nenhuma ação funciona se apenas poucas ações forem executadas. Usar:

```text
INVIABILIDADE CORROBORADA POR REPLAY
```

Para cada G3 revisar/executar:

- ação oracle de menor déficit;
- ação equal;
- ação prop;
- ações de referência necessárias;
- cross-check das 900/900 combinações já recomputadas.

Se nenhuma ação factível for encontrada pelo oracle e as melhores ações executadas ainda violarem:

```text
INFEASIBILITY CORROBORATED
```

Calcular:

\[
ICR =
\frac{
N(G3\ corroborados)
}{
N(G3\ executados)
}
\]

Registrar também D_min por replay quando possível.

---

## 9. G4 — causalidade URLLC

Objetivo: determinar se a cauda de deadline-miss URLLC responde causalmente à quantidade de recurso atribuída à URLLC.

Ações:

\[
a_u \in \{13,25,35\}
\]

Como esses pontos podem ficar fora do suporte observado, G4 é:

```text
teste causal de mecanismo
```

e não validação estatística do oracle offline.

Para cada janela, plotar:

\[
a_u
\]

versus:

```text
deadline-miss URLLC
latency p95
```

### Interpretação A

Se:

\[
a_u \uparrow
\Rightarrow
deadline\_miss \downarrow
\]

de forma consistente:

```text
CAUDA URLLC RESOURCE-DEPENDENT
```

### Interpretação B

Se aumentar a_u pouco altera a cauda:

```text
CAUDA PROVAVELMENTE DOMINADA POR SCHEDULER/TDD
```

---

## 10. Confusion matrix do oracle

Após G1–G3, construir:

| | Replay viável | Replay inviável/corroborado |
|---|---:|---:|
| Oracle viável | TP_feas | FP_feas |
| Oracle inviável | FN_feas / false-infeasible | TN_infeas |

Usar como classe positiva:

```text
INFEASIBLE
```

Calcular:

\[
precision_{inf}
\]

\[
recall_{inf}
\]

\[
F1_{inf}
\]

\[
accuracy
\]

e principalmente:

\[
false\_infeasible\_rate.
\]

---

## 11. Gate de confiança do oracle

O oracle é aprovado se:

1. G2 confirmar violações evitáveis em taxa alta;
2. G3 corroborar estados candidatos a inviabilidade;
3. false-infeasible for baixo;
4. erros se concentrarem em região marginal e forem documentáveis;
5. não houver inconsistência sistemática com os replays.

Se falhar:

```text
NÃO congelar P_inf
NÃO escolher L_O
NÃO implementar SQR-JC
```

Recalibrar primeiro.

---

## 12. Congelamento da porta de decisão

Definir uma única regra oficial para:

```text
FEASIBLE
INFEASIBLE
```

Documentar:

- porta principal;
- porta auxiliar;
- justificativa;
- sensibilidade mediana/p25;
- tratamento do G1.

Não alterar a porta depois da figura final sem versionar a mudança.

---

## 13. Recalcular o oracle completo

Somente após gate PASS.

Reprocessar todas as janelas da Etapa 2.3:

```text
equal-split
prop
```

para:

\[
\lambda \in \{2,4,5,6,7\}.
\]

Produzir por janela:

```text
joint_feasible
num_feasible_actions
avoidable_any
avoidable_embb
avoidable_urllc
avoidable_mmtc
min_total_deficit
best_feasible_action
best_degradation_action
```

---

## 14. Métricas finais

### Probabilidade de inviabilidade

\[
P_{inf}(\lambda)
=
P(\mathcal F_t=\emptyset)
\]

### Avoidable Violation Rate

\[
AVR_{arm}(\lambda)
=
P(
V_t=1
\mid
\mathcal F_t\neq\emptyset
)
\]

Calcular para equal, prop e por slice.

### Tamanho do conjunto factível

\[
|\mathcal F_t|
\]

Reportar mediana, p25, p75 e proporção de zero.

### Severidade mínima

Em estados inviáveis:

\[
D_{min}(t)
=
\min_{\mathbf a}
\sum_iD_i(\mathbf a)
\]

---

## 15. Separar três fenômenos

### A — má coordenação

```text
V = 1
F_t != vazio
```

= violação evitável.

### B — margem pequena

```text
F_t != vazio
|F_t| pequeno
```

= região marginal.

### C — inviabilidade real

```text
F_t = vazio
```

= alguma degradação é inevitável.

---

## 16. Tabela central

| λ | P_inf | AVR equal | AVR prop | |F_t| med | D_min | Regime |
|---:|---:|---:|---:|---:|---:|---|
| 2 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |

---

## 17. Tabela de validação

| Grupo | n | Confirmados | Falhas | Taxa | Leitura |
|---|---:|---:|---:|---:|---|
| G1 false-infeasible | 8 | | | | |
| G2 avoidable | 8 | | | | |
| G3 infeasible | 6 | | | | |
| G4 URLLC causal | 6 | | | | |

---

## 18. Figuras obrigatórias

1. `P_inf × λ`
2. `|F_t| × λ`
3. `AVR equal × prop`
4. `D_min × λ`
5. `a_u × deadline-miss/p95 URLLC`
6. mapa de estados viável/marginal/inviável
7. episódio temporal representativo

---

## 19. Escolha de L_N

Escolher L_N com:

\[
P_{inf}\approx0
\]

e:

\[
|\mathcal F_t|
\]

alto.

Também exigir:

- rádio saudável;
- SLAs atingíveis;
- ampla margem;
- AVR residual baixa.

---

## 20. Escolha de L_O

Escolher L_O na região:

\[
0<P_{inf}<1.
\]

O objetivo é:

```text
overload transitório
```

e não:

```text
colapso permanente
```

Não congelar automaticamente λ7.

λ5 ou λ6 podem representar melhor o overload transitório dependendo do P_inf e de |F_t|.

---

## 21. Decisão da arquitetura SQR-JC

### Caso 1 — AVR alto, P_inf baixo

Implementar primeiro:

```text
Joint Coordination
+
preservação dos mínimos seguros
```

### Caso 2 — P_inf relevante

Implementar:

```text
Feasibility Awareness
+
Graceful Degradation
+
Anti-Windup
```

### Caso 3 — ambos

Implementar SQR-JC completo:

```text
SQR
+
Feasibility Detection
+
Joint Coordination
+
Graceful Degradation
+
Anti-Windup
```

Regra:

```text
oracle validado
      ↓
AVR + P_inf + D_min
      ↓
arquitetura SQR-JC
```

---

## 22. Atualizar documentação

Após adjudicação:

- marcar II-b concluída;
- registrar II-c;
- registrar gate PASS/STOP;
- registrar porta congelada;
- registrar L_N;
- registrar L_O;
- registrar arquitetura escolhida;
- registrar limitações.

Atualizar assumptions/register com os próximos IDs disponíveis para:

```text
seleção G1–G4
uso de p25 em G1
resultado do gate executável
porta final
L_N
L_O
decisão SQR-JC
```

---

## 23. Artefatos esperados

```text
results/sqr_characterization_round2/oracle/
├── replay/
│   ├── raw/
│   ├── validated/
│   └── oracle_replay_results.csv
├── adjudication/
│   ├── confusion_matrix.csv
│   ├── group_summary.csv
│   ├── g4_urllc_curve.csv
│   └── adjudication_notes.md
├── final/
│   ├── oracle_windows_final.csv
│   ├── oracle_by_lambda_final.csv
│   ├── oracle_by_slice_final.csv
│   ├── feasibility_boundary.csv
│   └── ln_lo_selection.md
└── plots/
    ├── fig_pinf.png
    ├── fig_fset.png
    ├── fig_avr.png
    ├── fig_dmin.png
    └── fig_g4_urllc.png
```

---

## 24. Relatório final obrigatório

Criar:

```text
relatorio_oracle_2_4_final.md
```

Estrutura:

```text
1. objetivo
2. protocolo
3. validação determinística do replay
4. G1
5. G2
6. G3
7. G4
8. confusion matrix
9. gate do oracle
10. P_inf
11. AVR
12. |F_t|
13. D_min
14. fronteira
15. L_N
16. L_O
17. decisão SQR-JC
18. limitações
19. conclusão
```

---

## 25. Linguagem obrigatória

Usar:

```text
"violação evitável confirmada por replay"
```

somente em G2 quando a ação alternativa realmente satisfizer todos os SLAs.

Usar:

```text
"inviabilidade corroborada por replay"
```

para G3.

Evitar:

```text
"o replay provou matematicamente que nenhuma ação funciona"
```

A prova do vazio de F_t vem do oracle enumerativo validado.

---

## 26. Limitações a registrar

- G1 usa testemunha p25;
- G4 contém ações fora do suporte;
- replay é validação amostral;
- oracle offline depende da qualidade do modelo;
- SLAs heterogêneos;
- deadline URLLC é experimental;
- mMTC Perfil A permanece leve;
- P_inf é condicionado ao cenário adotado.

---

## 27. Gate final da Etapa 2.4

### PASS

Se:

- replay determinístico confirmado;
- false-infeasible baixo;
- G2 confirma evitabilidade;
- G3 corrobora inviabilidade;
- oracle offline e executável concordam suficientemente;
- fronteira por carga é consistente.

Então:

```text
ETAPA 2.4 = PASS
```

e seguir para especificação/implementação do SQR-JC.

### STOP

Se:

- false-infeasible alto;
- G2 não confirma ações alternativas;
- G3 contradiz sistematicamente o oracle;
- fronteira depende excessivamente da porta;
- modelo extrapola demais;
- replay não reproduz o estado.

Então:

```text
ETAPA 2.4 = STOP
```

e recalibrar o oracle.

---

## 28. Próximo passo após PASS

Somente após PASS:

1. congelar L_N;
2. congelar L_O;
3. congelar arquitetura SQR-JC;
4. escrever especificação algorítmica;
5. implementar;
6. criar episódio transitório:

```text
L_N -> L_O -> L_N
```

7. comparar:

```text
SQR original
prop
SQR-JC
oracle
```

---

## 29. Pergunta científica a responder

A Etapa 2.4-II-c deve permitir afirmar quantitativamente:

> **Em quais estados as violações observadas poderiam ter sido evitadas por uma alocação melhor, e em quais estados a capacidade da célula não permitia satisfazer simultaneamente todos os SLAs?**

---

## 30. Comando resumido para o agente

```text
Concluir os 28 replays em andamento sem alterar protocolo, binário,
SLAs ou seleção das amostras.

Depois executar a Etapa 2.4-II-c:

1. adjudicar G1–G4;
2. construir confusion matrix do oracle com foco em false-infeasible;
3. gerar curva causal URLLC do G4;
4. aplicar o gate de confiança;
5. se PASS, recalcular o oracle em todas as janelas da 2.3;
6. produzir P_inf, AVR, |F_t| e D_min;
7. congelar a fronteira de viabilidade;
8. definir L_N e L_O a partir dos dados;
9. decidir a arquitetura final do SQR-JC;
10. gerar relatório final e atualizar PLANO/register.

Não implementar o SQR-JC antes do PASS da Etapa 2.4.
```
