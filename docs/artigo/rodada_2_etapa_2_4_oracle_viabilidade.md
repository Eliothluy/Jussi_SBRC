# Rodada 2 — Etapa 2.4: Oracle de Viabilidade Conjunta

**Objetivo:** classificar, por janela de decisão, se o estado observado da rede ainda admite pelo menos uma alocação de recursos capaz de satisfazer simultaneamente os SLAs de eMBB, URLLC e mMTC, separando formalmente violações evitáveis de estados conjuntamente inviáveis.

**Pré-condição:** Etapa 2.3 concluída, com dois braços experimentais (`equal-split` e `prop`) sob a mesma infraestrutura, mesmos SLAs, mesmos perfis de tráfego e mesmas cargas.

---

## 1. Motivação da Etapa 2.4

A Etapa 2.3 mostrou que a política de alocação altera fortemente a distribuição das violações:

- `equal-split` deixa recursos ociosos e produz muitas violações eMBB;
- `prop` reutiliza essa capacidade e reduz fortemente as violações eMBB;
- porém, sob maior pressão, `prop` aumenta a cauda de latência da URLLC;
- mMTC permanece praticamente sem pressão de capacidade no Perfil A.

Portanto, a Etapa 2.3 fornece **forte evidência de que parte das violações depende da coordenação**, mas não permite afirmar formalmente quais violações são evitáveis.

A pergunta central desta etapa é:

> **Para o estado observado em uma janela \(t\), existe alguma alocação válida de recursos que satisfaz simultaneamente todos os SLAs?**

Essa pergunta define a fronteira de viabilidade.

---

## 2. Definição formal

Considere o vetor de alocação:

\[
\mathbf{a}_t =
\left(
a_t^{eMBB},
a_t^{URLLC},
a_t^{mMTC}
\right)
\]

com:

\[
a_t^{eMBB}+a_t^{URLLC}+a_t^{mMTC}\le C
\]

onde, no cenário atual:

\[
C=53.
\]

Defina o conjunto de ações admissíveis:

\[
\mathcal A=\left\{\mathbf a: a_i\ge0,\;\sum_i a_i\le C\right\}.
\]

Para cada estado observado \(z_t\), defina:

\[
\mathcal F_t=
\left\{
\mathbf a\in\mathcal A:
SLA_{eMBB}(\mathbf a,z_t)=1,\;
SLA_{URLLC}(\mathbf a,z_t)=1,\;
SLA_{mMTC}(\mathbf a,z_t)=1
\right\}.
\]

### Estado conjuntamente viável

Se:

\[
\mathcal F_t\neq\emptyset,
\]

existe pelo menos uma alocação capaz de atender todos os SLAs simultaneamente.

Classificação:

```text
JOINTLY_FEASIBLE
```

### Estado conjuntamente inviável

Se:

\[
\mathcal F_t=\emptyset,
\]

nenhuma alocação admissível consegue atender todos os SLAs ao mesmo tempo.

Classificação:

```text
JOINTLY_INFEASIBLE
```

---

## 3. Violação evitável e violação em estado inviável

Se o controlador realmente executado violou algum SLA:

\[
V_t=1
\]

mas:

\[
\mathcal F_t\neq\emptyset,
\]

classificar como:

\[
\boxed{\text{AVOIDABLE}}
\]

Interpretação: a capacidade e o estado permitiam uma solução válida, mas a política usada não encontrou uma alocação conjunta satisfatória.

Se:

\[
V_t=1
\]

e:

\[
\mathcal F_t=\emptyset,
\]

classificar como:

\[
\boxed{\text{UNAVOIDABLE AT SYSTEM LEVEL}}
\]

Interpretação: não existia uma alocação capaz de preservar simultaneamente todos os SLAs naquela janela.

### Cuidado conceitual

Quando \(\mathcal F_t=\emptyset\), isso **não significa que a violação de cada slice individualmente era inevitável**.

Pode existir, por exemplo:

```text
ação A -> satisfaz URLLC + mMTC, viola eMBB
ação B -> satisfaz eMBB + mMTC, viola URLLC
```

Logo, em estado inviável, também é necessário calcular **qual é a melhor degradação possível**.

---

## 4. Estado \(z_t\) usado pelo oracle

O oracle não deve usar apenas \(\lambda\).

A Etapa 2.0 mostrou que o comportamento é episódico e dependente do estado instantâneo.

Cada janela deve registrar pelo menos:

```text
time
seed
lambda_total
arm
N_embb
N_urllc
N_mmtc
offered_load_embb
offered_load_urllc
offered_load_mmtc
queue_embb
queue_urllc
queue_mmtc
throughput_embb
latency_urllc
pdr_urllc
pdr_mmtc
snr
cqi
prb_requested
prb_scheduled
prb_executed
prb_idle
util_active_dl
util_all_dl
```

Se alguma variável não estiver diretamente disponível, registrar explicitamente a proxy utilizada.

---

## 5. Estratégia do oracle

A implementação deve usar dois níveis:

```text
Oracle offline (principal)
        +
Validação executável por amostragem
```

O oracle offline permite classificar todas as janelas da campanha. A validação executável serve para verificar se a classificação offline é confiável perto da fronteira.

---

## 6. Oracle offline — função principal

O oracle offline deve estimar, para cada estado \(z_t\), os KPIs resultantes sob várias alocações candidatas.

Formalmente:

\[
\hat y_i=f_i(z_t,\mathbf a)
\]

onde \(f_i\) estima o KPI da slice \(i\).

Exemplos:

\[
\hat y_{eMBB}=f_{eMBB}(z_t,\mathbf a)
\]

\[
\hat y_{URLLC}^{lat}=f_{URLLC}^{lat}(z_t,\mathbf a)
\]

\[
\hat y_{mMTC}^{PDR}=f_{mMTC}^{PDR}(z_t,\mathbf a).
\]

O modelo deve documentar:

- entradas/features;
- target por KPI;
- divisão treino/validação;
- seeds;
- erro global;
- erro próximo aos limiares dos SLAs;
- região de suporte dos dados.

### Regra principal

Para cada janela:

```text
state = z_t
F_t = []

for action in admissible_actions:
    predict KPIs under action

    if all SLAs satisfied:
        F_t.append(action)

joint_feasible = len(F_t) > 0
num_feasible_actions = len(F_t)
```

---

## 7. Espaço de ações

Com três slices e capacidade \(C=53\), enumerar inicialmente todas as combinações discretas:

\[
a_{eMBB},a_{URLLC},a_{mMTC}\in\{0,\ldots,C\}
\]

sujeitas a:

\[
a_{eMBB}+a_{URLLC}+a_{mMTC}\le C.
\]

Preferir busca exaustiva com passo de 1 RBG/PRB.

Se o custo for alto, medir antes o tempo da enumeração e somente depois considerar discretização maior.

```text
passo = 1 RBG  # preferencial
passo = 2 RBG  # somente se necessário
```

Não usar passos maiores sem avaliar a perda de resolução da fronteira.

---

## 8. SLAs congelados

Usar exatamente os mesmos SLAs das Etapas 2.2 e 2.3.

### eMBB

\[
throughput_{UE}\ge3.5\text{ Mbps}
\]

Aplicar exatamente a mesma regra usada na análise da 2.3.

### URLLC

Deadline:

\[
latency\le10\text{ ms}
\]

Manter a mesma regra de deadline-miss utilizada anteriormente.

**Não alterar scheduler nem deadline nesta etapa.**

### mMTC

Manter:

\[
PDR\ge95\%
\]

ou o valor efetivamente congelado no registro R38.

Não recalibrar SLAs durante a execução do oracle.

---

## 9. Satisfação e déficit normalizados

### Throughput

\[
S_i=\min\left(1,\frac{y_i}{\Lambda_i}\right)
\]

\[
D_i=\frac{[\Lambda_i-y_i]_+}{\Lambda_i}.
\]

### Latência

\[
S_i^{lat}=\min\left(1,\frac{\Lambda_i}{y_i}\right)
\]

\[
D_i^{lat}=\frac{[y_i-\Lambda_i]_+}{\Lambda_i}.
\]

### PDR

\[
S_i^{PDR}=\min\left(1,\frac{y_i}{\Lambda_i}\right)
\]

\[
D_i^{PDR}=\frac{[\Lambda_i-y_i]_+}{\Lambda_i}.
\]

---

## 10. Melhor ação quando \(\mathcal F_t=\emptyset\)

Se nenhuma ação satisfizer todos os SLAs, calcular:

\[
\mathbf a_t^*=\arg\min_{\mathbf a\in\mathcal A}\sum_i w_iD_i(\mathbf a).
\]

Registrar:

```text
best_degradation_action
min_total_deficit
deficit_embb
deficit_urllc
deficit_mmtc
```

Primeiro calcular também a versão neutra:

\[
w_i=1.
\]

Qualquer priorização por criticidade deverá ser reportada separadamente para não confundir caracterização de viabilidade com política de prioridade.

---

## 11. Trade-off eMBB × URLLC

A Etapa 2.3 mostrou:

```text
equal-split -> protege relativamente URLLC, prejudica eMBB
prop        -> recupera eMBB, piora a cauda URLLC sob carga
```

O oracle deve procurar explicitamente se existe uma terceira região de alocação que satisfaça ambos.

Para janelas representativas, registrar:

```text
allocation
throughput_embb
deadline_miss_urllc
pdr_mmtc
```

Isso permitirá visualizar uma possível fronteira de Pareto.

---

## 12. Métricas principais

### 12.1. Probabilidade de inviabilidade

\[
P_{inf}(\lambda)=P(\mathcal F_t=\emptyset).
\]

Esta é a principal métrica da fronteira.

### 12.2. Probabilidade de viabilidade

\[
P_{feas}(\lambda)=1-P_{inf}(\lambda).
\]

### 12.3. Avoidable Violation Rate

\[
AVR=P(V_t=1\mid\mathcal F_t\neq\emptyset).
\]

Calcular:

- global;
- por slice;
- por \(\lambda\);
- por braço;
- por seed.

### 12.4. Violation rate em estados inviáveis

\[
UVR=P(V_t=1\mid\mathcal F_t=\emptyset).
\]

### 12.5. Tamanho do conjunto factível

\[
|\mathcal F_t|.
\]

Interpretação:

```text
grande  -> muita folga
pequeno -> região marginal
zero    -> inviável
```

### 12.6. Severidade mínima possível

\[
D_{min}(t)=\min_{\mathbf a\in\mathcal A}\sum_iD_i(\mathbf a).
\]

Essa métrica mede o menor dano possível quando o sistema já está inviável.

---

## 13. Métricas por slice

Para cada janela, registrar:

```text
avoidable_embb
avoidable_urllc
avoidable_mmtc

min_deficit_embb
min_deficit_urllc
min_deficit_mmtc
```

Em estado inviável, determinar também se cada slice ainda poderia ser individualmente preservada:

```text
embb_individually_saveable
urllc_individually_saveable
mmtc_individually_saveable
```

Isso evita concluir incorretamente que toda violação individual era inevitável apenas porque \(\mathcal F_t=\emptyset\).

---

## 14. Equal, prop e oracle

`equal-split` e `prop` são testemunhas/baselines, não oracle.

Para cada janela, registrar quando possível:

```text
equal_action
prop_action
oracle_feasible_action
oracle_best_action
```

### Caso A

```text
equal viola
prop satisfaz
oracle confirma F_t != vazio
```

Interpretação: evidência direta de violação evitável do equal.

### Caso B

```text
equal viola
prop viola
oracle encontra ação satisfatória
```

Interpretação: ambos falharam, mas o estado era viável.

Esse caso é especialmente importante para justificar SQR-JC.

### Caso C

```text
equal viola
prop viola
oracle encontra F_t vazio
```

Interpretação: estado conjuntamente inviável.

### Caso D

```text
equal satisfaz
prop viola
oracle encontra F_t != vazio
```

Interpretação: o prop deslocou recursos de forma inadequada e criou uma violação evitável, particularmente relevante para URLLC.

---

## 15. Validação executável do oracle

O oracle offline não deve ser aceito sem validação.

Selecionar amostra representativa de janelas em:

```text
lambda = 2
lambda = 4
lambda = 5
lambda = 6
lambda = 7
```

Priorizar:

- equal viola e prop não;
- prop viola URLLC;
- ambos violam;
- oracle classifica como inviável;
- \(|F_t|\) pequeno;
- estados próximos ao SLA.

### Replay ideal

```text
snapshot state at t
for each candidate action:
    restore snapshot
    execute action
    observe KPI
```

Se snapshot não estiver disponível, usar alternativa reproduzível:

1. traces congelados;
2. replay determinístico de arrivals/sessions;
3. reconstrução por seed e estado;
4. microepisódios reproduzíveis;
5. replay parcial em subconjunto de janelas.

Documentar explicitamente qualquer limitação.

---

## 16. Validação do modelo offline

Para cada janela executável, comparar:

```text
predicted feasible/infeasible
executed feasible/infeasible
```

Construir:

```text
TP
TN
FP
FN
```

Calcular pelo menos:

\[
accuracy
\]

\[
precision_{infeasible}
\]

\[
recall_{infeasible}
\]

\[
F1_{infeasible}.
\]

Dar atenção especial a **false infeasible**, pois classificar uma janela viável como inviável pode inflar artificialmente a contribuição.

### Gate de confiança

Não congelar a fronteira se:

- erro de KPI for alto perto do SLA;
- a classificação mudar muito entre modelos;
- muitas ações avaliadas estiverem fora do suporte dos dados;
- a validação executável contradizer frequentemente o oracle offline.

Nesses casos:

```text
STOP
```

e refinar modelo/dataset.

---

## 17. Hipótese de regimes a testar

A 2.3 sugere provisoriamente:

```text
lambda = 2 -> predominantemente viável
lambda = 4 -> marginal candidato
lambda = 5 -> marginal / início de inviabilidade candidato
lambda = 6 -> overload moderado candidato
lambda = 7 -> stress candidato
```

Essa classificação **não está congelada**.

O oracle deve poder refutá-la.

---

## 18. Fronteira de viabilidade

A fronteira principal deve ser expressa como:

\[
P_{inf}(\lambda)
\]

em vez de um único valor rígido de \(\lambda\), porque o fenômeno é estocástico.

Preferir:

```text
P_inf ≈ 0       -> regime viável
0 < P_inf < 1   -> região de fronteira / overload transitório
P_inf ≈ 1       -> overload severo
```

### Classificação final

**Normal**

\[
P_{inf}\approx0
\]

com \(|F_t|\) alto.

**Marginal**

\[
P_{inf}\approx0
\]

mas \(|F_t|\) baixo.

**Overload transitório**

\[
0<P_{inf}<1.
\]

Essa é a região principal de interesse científico.

**Overload severo**

\[
P_{inf}\rightarrow1.
\]

Usar apenas como limite de stress.

---

## 19. CSV principal do oracle

Criar:

```csv
seed,
time,
lambda_total,
arm,
n_embb,
n_urllc,
n_mmtc,
queue_embb,
queue_urllc,
queue_mmtc,
util_active_dl,
util_all_dl,
observed_embb_violation,
observed_urllc_violation,
observed_mmtc_violation,
observed_any_violation,
joint_feasible,
num_feasible_actions,
avoidable_any,
avoidable_embb,
avoidable_urllc,
avoidable_mmtc,
embb_individually_saveable,
urllc_individually_saveable,
mmtc_individually_saveable,
best_feasible_embb,
best_feasible_urllc,
best_feasible_mmtc,
best_degradation_embb,
best_degradation_urllc,
best_degradation_mmtc,
min_total_deficit,
min_deficit_embb,
min_deficit_urllc,
min_deficit_mmtc
```

---

## 20. Tabelas principais

### Tabela geral

| \(\lambda\) | \(P_{inf}\) | \(P_{feas}\) | AVR equal | AVR prop | \(|F_t|\) mediano | \(D_{min}\) | Regime |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |
| 6 | | | | | | | |
| 7 | | | | | | | |

### Tabela por slice

| \(\lambda\) | AVR eMBB | AVR URLLC | AVR mMTC | saveable eMBB | saveable URLLC | saveable mMTC |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |

---

## 21. Figuras obrigatórias

### Figura 1 — \(P_{inf}\) × carga

Eixo x:

\[
\lambda_{total}
\]

Eixo y:

\[
P(\mathcal F_t=\emptyset).
\]

### Figura 2 — \(|F_t|\) × carga

Mostrar mediana e p25/p75.

### Figura 3 — AVR × carga

Separar:

```text
equal-split
prop
```

Se possível, também por slice.

### Figura 4 — severidade mínima

Plotar:

\[
D_{min}
\]

nas janelas inviáveis.

### Figura 5 — mapa de estados

Exemplos:

```text
N_total × queue_total
```

ou:

```text
N_embb × N_urllc
```

com cor:

```text
feasible
marginal
infeasible
```

### Figura 6 — trade-off eMBB × URLLC

Para janelas selecionadas:

```text
x = throughput eMBB
y = deadline miss URLLC
```

um ponto por alocação candidata.

Destacar equal, prop e oracle.

### Figura 7 — episódio temporal

Mostrar:

```text
N(t)
queue(t)
allocation(t)
SLA(t)
joint_feasible(t)
|F_t|
```

---

## 22. Sanity checks do oracle

### SC1 — tendência com carga

Esperado:

\[
P_{inf}(\lambda)
\]

tende a crescer com a carga.

Não exigir monotonicidade perfeita por causa da estocasticidade.

### SC2 — baixa carga

Em \(\lambda=2\), espera-se:

\[
P_{inf}\approx0.
\]

Se for alto, revisar oracle/SLA/cenário.

### SC3 — coerência com prop

Se `prop` satisfaz uma janela em que equal viola, o oracle deve classificar a janela como viável.

Caso contrário, investigar imediatamente a inconsistência.

### SC4 — rádio

Confirmar que janelas classificadas como inviáveis não são dominadas por:

```text
RLF
SNR colapsado
zero-delivery de PHY
```

A fronteira pretendida deve ser de recurso/demanda, não cobertura.

---

## 23. Hipóteses falsificáveis

### H2.4-1

A probabilidade de inviabilidade tende a crescer com a carga:

\[
P_{inf}(2)<P_{inf}(4)<P_{inf}(5)<P_{inf}(6)<P_{inf}(7)
\]

como tendência, não necessariamente de forma estritamente monotônica em cada seed.

### H2.4-2

Existe uma região em que coexistem:

\[
AVR>0
\]

e:

\[
0<P_{inf}<1.
\]

Essa é a região ideal para justificar SQR-JC.

### H2.4-3

`prop` reduz AVR eMBB em relação ao equal-split, mas não necessariamente AVR URLLC.

### H2.4-4

Em alta carga:

\[
D_{min}>0
\]

em parte significativa das janelas.

Isso indicará que nenhuma política pode eliminar completamente as violações nesses estados.

---

## 24. Decisão sobre o SQR-JC após o oracle

### Caso A — AVR alto e \(P_{inf}\) baixo

Priorizar:

```text
Joint Coordination
+
preservação de mínimos seguros
```

### Caso B — \(P_{inf}\) alto

Priorizar:

```text
Feasibility Awareness
+
Graceful Degradation
+
Anti-Windup
```

### Caso C — ambos relevantes

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

---

## 25. Gate de go/no-go do SQR-JC

O SQR-JC deve avançar como contribuição principal se pelo menos uma destas condições ocorrer:

1. existe parcela relevante de violações com \(\mathcal F_t\neq\emptyset\);
2. existem estados com \(\mathcal F_t=\emptyset\) durante overload transitório;
3. equal e prop deixam espaço mensurável para uma terceira política conjunta;
4. há necessidade clara de distinguir erro de decisão de inviabilidade física.

Reavaliar a proposta se:

- \(P_{inf}\approx0\) em toda a faixa;
- prop praticamente coincide com oracle;
- AVR residual é desprezível;
- não existe região transitória;
- as violações forem novamente explicadas principalmente pela camada física;
- o oracle não puder ser validado com confiança suficiente.

---

## 26. Escolha de \(L_N\) e \(L_O\)

Somente após o oracle.

### \(L_N\)

Escolher carga com:

\[
P_{inf}\approx0
\]

e grande \(|F_t|\).

### \(L_O\)

Escolher carga moderada com:

\[
0<P_{inf}<1.
\]

Evitar carga em que:

\[
P_{inf}\approx1.
\]

O objetivo é estudar overload transitório, não colapso permanente.

---

## 27. Episódio transitório futuro

Depois da 2.4:

```text
NORMAL
  ↓
MARGINAL
  ↓
OVERLOAD TRANSITÓRIO
  ↓
MARGINAL
  ↓
NORMAL
```

O episódio deve usar \(L_N\) e \(L_O\) obtidos empiricamente.

---

## 28. Artefatos esperados

```text
results/sqr_characterization_round2/oracle/
├── models/
├── predictions/
├── executable_validation/
├── plots/
├── tables/
└── reports/
```

Arquivos:

```text
oracle_windows.csv
oracle_actions.csv
oracle_by_lambda.csv
oracle_by_slice.csv
oracle_validation.csv
oracle_tradeoff_samples.csv
fig_oracle_pinf.png
fig_oracle_fset.png
fig_oracle_avr.png
fig_oracle_deficit.png
fig_oracle_state_map.png
fig_oracle_pareto.png
relatorio_oracle_2_4.md
```

---

## 29. Checklist operacional

### Preparação

- [ ] 30/30 runs da 2.3 íntegros.
- [ ] KPM disponível.
- [ ] `packet_delays` disponível.
- [ ] \(N(t)\) extraído por slice.
- [ ] filas extraídas.
- [ ] ações equal/prop identificadas.
- [ ] SLAs congelados.

### Modelo offline

- [ ] Dataset construído.
- [ ] Features documentadas.
- [ ] Targets documentados.
- [ ] Split treino/validação definido.
- [ ] Métricas de erro calculadas.
- [ ] Erro perto dos limiares de SLA analisado.

### Busca

- [ ] Simplex de ações gerado.
- [ ] Todas as ações válidas avaliadas.
- [ ] \(\mathcal F_t\) calculado.
- [ ] \(|F_t|\) calculado.
- [ ] ação factível registrada.
- [ ] melhor ação de degradação registrada.

### Classificação

- [ ] `joint_feasible`.
- [ ] `avoidable_any`.
- [ ] evitabilidade por slice.
- [ ] \(P_{inf}\).
- [ ] AVR.
- [ ] \(D_{min}\).

### Validação

- [ ] Janelas críticas amostradas.
- [ ] Replay executado.
- [ ] Confusion matrix calculada.
- [ ] Erros investigados.
- [ ] Gate de confiança aprovado.

### Saída

- [ ] Tabela principal.
- [ ] Tabela por slice.
- [ ] Figuras 1–7.
- [ ] \(L_N\) definido.
- [ ] \(L_O\) definido.
- [ ] decisão SQR-JC registrada.

---

## 30. Critério de encerramento da Etapa 2.4

A Etapa 2.4 só estará encerrada quando for possível responder quantitativamente:

1. Qual é \(P_{inf}(\lambda)\)?
2. Onde \(|\mathcal F_t|\) começa a colapsar?
3. Qual parcela das violações do equal era evitável?
4. Qual parcela das violações do prop ainda era evitável?
5. Em quais estados nenhuma política consegue satisfazer todos os SLAs?
6. Qual é a menor severidade possível nesses estados?
7. Qual slice ainda pode ser preservada em cada estado inviável?
8. Qual carga representa melhor o regime normal?
9. Qual carga representa melhor overload transitório?
10. O SQR-JC é justificado pelos dados?

---

## 31. Resultado científico esperado

A Etapa 2.4 deve transformar a narrativa:

> “uma política melhor reduz violações”

em:

> **“quantificamos, estado a estado, quando uma violação decorre de uma decisão de coordenação subótima e quando decorre da inexistência de qualquer alocação capaz de satisfazer simultaneamente todos os SLAs.”**

Essa distinção é o fundamento da contribuição baseada em **feasibility-aware joint coordination**.

---

## 32. Regra metodológica principal

Até o encerramento do oracle:

```text
equal x prop = evidência de espaço de coordenação
```

e **não**:

```text
equal x prop = evitável x inevitável
```

Somente a análise de:

\[
\mathcal F_t
\]

permite a classificação formal.

---

## 33. Prioridade de execução

```text
1. construir dataset por janela
2. treinar e validar modelos de KPI
3. gerar simplex de ações
4. calcular F_t
5. gerar P_inf, AVR e |F_t|
6. identificar janelas críticas
7. validar por replay executável
8. fechar fronteira
9. definir L_N / L_O
10. decidir arquitetura do SQR-JC
```

**Não implementar o SQR-JC antes de concluir os itens 1–8.**
