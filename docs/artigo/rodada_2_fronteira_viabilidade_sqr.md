# Rodada 2 — Caracterização da Fronteira de Viabilidade com Slices Heterogêneos

## 1. Objetivo da Rodada 2

A Rodada 2 tem como objetivo transformar os indícios obtidos na Rodada 1b em uma **caracterização formal da fronteira de viabilidade** do cenário, agora considerando slices heterogêneos:

- **eMBB**;
- **URLLC**;
- **mMTC**.

A pergunta central é:

> **Em cada estado da rede, existe pelo menos uma alocação de recursos capaz de satisfazer simultaneamente todos os SLAs?**

A Rodada 2 deve permitir separar:

\[
\text{violação evitável}
\]

de

\[
\text{violação inevitável}.
\]

Essa distinção é a base experimental para decidir se a proposta **SQR-JC** é justificável e quais componentes dela realmente precisam ser implementados.

---

# 2. O que já foi estabelecido na Rodada 1b

A Rodada 1b já forneceu evidência de que a degradação observada pode ser produzida por **pressão de demanda**, sem degradar artificialmente a camada de rádio.

Resultados principais:

| \(\lambda\) (UE/min) | Violação UE-tempo | Utilização DL ativa | p5 throughput/UE |
|---:|---:|---:|---:|
| 2 | 2,0% | 41% | 3,2 Mbps |
| 4 | 5,6% | 57% | 1,9 Mbps |
| 5 | 14,0% | 68% | 1,4 Mbps |
| 6 | 18,3% | 76% | 0,8 Mbps |
| 7 | 24,1% | 84% | 0,6 Mbps |
| 8 | 35,8% | 88% | 0,04 Mbps |

Outros achados:

- RLF = 0;
- SNR dos UEs que violam é semelhante ao dos que não violam;
- a degradação cresce progressivamente com a carga;
- o p5 do throughput degrada antes da mediana;
- \(\lambda=4\)–5 representa região candidata a violações evitáveis;
- \(\lambda=7\) representa candidato a overload moderado;
- \(\lambda=8\) representa um limite superior de estresse severo.

## Importante

Essas classificações ainda são **candidatas**.

Ainda não é permitido afirmar:

\[
\lambda=4\text{–}5
\Rightarrow
\text{violação evitável}
\]

ou:

\[
\lambda=7\text{–}8
\Rightarrow
\text{violação inevitável}.
\]

Essa classificação exige o **oracle de viabilidade**.

---

# 3. Estrutura geral da Rodada 2

A Rodada 2 será dividida em seis etapas:

```text
2.0 — Estabilidade temporal
       ↓
2.1 — Instrumentação de KPIs heterogêneos
       ↓
2.2 — Validação em baixa carga
       ↓
2.3 — Sweep heterogêneo
       ↓
2.4 — Oracle de viabilidade
       ↓
2.5 — Episódio transitório
```

Somente depois dessas etapas deverá ser tomada a decisão de implementação do SQR-JC.

---

# 4. Etapa 2.0 — Estabilidade temporal

## 4.1. Motivação

Na Rodada 1b foi observado que a taxa de violação cresce com o tempo de execução.

Como as sessões possuem duração média de aproximadamente 60 s, os primeiros instantes do experimento podem estar contaminados pelo processo de enchimento da célula.

Antes de introduzir a heterogeneidade, verificar se os regimes identificados permanecem estáveis após um período de warm-up.

## 4.2. Pontos a testar

Executar novamente:

\[
\lambda \in \{2,5,7\}
\]

com:

- mesmo rádio congelado;
- mesmo raio;
- mesma potência;
- mesmo canal;
- mesma política/scheduler;
- mesmas condições da Rodada 1b;
- múltiplas seeds.

Aumentar a duração total do experimento.

Sugestão inicial:

```text
T_total = 600 s
```

Separar:

```text
0–180 s      → warm-up
180–600 s    → região principal de medição
```

Os valores podem ser ajustados de acordo com custo computacional e comportamento observado.

## 4.3. Métricas

Comparar:

- taxa de violação;
- throughput mediano;
- p5/p10/p25;
- número médio de UEs ativos;
- utilização de recursos;
- tamanho das filas;
- SNR;
- RLF;
- recursos solicitados;
- recursos programados;
- recursos executados;
- recursos ociosos.

## 4.4. Critério de aprovação

A etapa é aprovada se a ordenação permanecer aproximadamente:

```text
λ2 → regime leve/normal
λ5 → regime marginal
λ7 → alta carga / overload candidato
```

e se as métricas após o warm-up apresentarem comportamento suficientemente estável.

Se a taxa de violação continuar crescendo sem estabilização, revisar:

- duração total;
- processo de chegada;
- duração das sessões;
- definição de \(L_0\);
- método de amostragem.

---

# 5. Etapa 2.1 — Instrumentação dos slices heterogêneos

## 5.1. Objetivo

Modificar a instrumentação atual para permitir o tratamento de SLAs distintos por tipo de serviço.

Os três slices deverão possuir KPIs diferentes.

---

# 6. KPIs por slice

## eMBB

KPI principal:

\[
\text{throughput}
\]

Sugestões de métricas:

- throughput por UE;
- p5/p10;
- throughput médio;
- packet loss;
- fila.

## URLLC

KPIs principais:

\[
\text{latência}
\]

e:

\[
\text{confiabilidade / PDR}.
\]

Registrar:

- latency mean;
- latency p95;
- latency p99;
- deadline miss ratio;
- packet delivery ratio;
- packet loss;
- queue delay;
- quantidade de pacotes fora do deadline.

## mMTC

KPI principal deve refletir atendimento/conectividade em massa.

Possíveis métricas:

- PDR;
- proporção de UEs atendidos;
- taxa de sucesso de entrega;
- throughput agregado;
- atraso, se relevante ao caso de uso.

---

# 7. Requisito de KPM

A instrumentação atual precisa ser estendida para incluir pelo menos:

```text
latency
pdr
deadline_miss
```

além das métricas já disponíveis:

```text
throughput
packet_loss
buffer
snr
resource_usage
```

Criar métricas por:

- UE;
- slice;
- janela de decisão.

---

# 8. SLAs

Os SLAs definitivos devem ser definidos com base em fonte da literatura ou no caso de uso adotado.

Criar uma tabela consolidada:

| Slice | KPI crítico | SLA | Sentido | Prioridade |
|---|---|---|---|---|
| eMBB | throughput | A definir | maior é melhor | média |
| URLLC | latência | A definir | menor é melhor | alta |
| URLLC | PDR/reliability | A definir | maior é melhor | alta |
| mMTC | PDR/atendimento | A definir | maior é melhor | conforme caso |

Não seguir para o sweep definitivo sem fechar essa tabela.

---

# 9. Normalização dos SLAs

Como os KPIs possuem unidades diferentes, criar uma métrica normalizada de satisfação.

Para throughput:

\[
S_i(t)
=
\min\left(
1,
\frac{y_i(t)}{\Lambda_i}
\right).
\]

Para latência:

\[
S_i^{lat}(t)
=
\min\left(
1,
\frac{\Lambda_i}{y_i(t)}
\right).
\]

Assim:

\[
S_i(t)=1
\]

significa SLA satisfeito ou superado.

Valores menores que 1 representam déficit.

---

# 10. Severidade da violação

Para throughput:

\[
D_i(t)
=
\frac{
[\Lambda_i-y_i(t)]_+
}{
\Lambda_i
}.
\]

Para latência:

\[
D_i^{lat}(t)
=
\frac{
[y_i(t)-\Lambda_i]_+
}{
\Lambda_i
}.
\]

Para PDR/reliability:

\[
D_i^{pdr}(t)
=
\frac{
[\Lambda_i-y_i(t)]_+
}{
\Lambda_i
}.
\]

Essa métrica será necessária posteriormente para avaliar degradação controlada.

---

# 11. Etapa 2.2 — Sanity check em baixa carga

## 11.1. Objetivo

Antes de procurar congestionamento, demonstrar que os três SLAs são atingíveis em condições de baixa demanda.

Isso evita confundir:

- erro de implementação;
- SLA impossível;
- problema de rádio;

com:

- falta de recursos.

## 11.2. Condição esperada

Em baixa carga:

\[
SLA_{eMBB}=OK
\]

\[
SLA_{URLLC}=OK
\]

\[
SLA_{mMTC}=OK.
\]

Além disso:

- RLF próximo de zero;
- SINR operacional;
- baixa fila;
- alta folga de recursos;
- PDR elevado;
- latência dentro do esperado.

## 11.3. Gate

Se qualquer slice violar de forma sistemática em baixa carga:

```text
STOP
```

Não prosseguir para a caracterização.

Investigar antes:

- SLA;
- KPI;
- tráfego;
- scheduler;
- camada física;
- cálculo da métrica;
- janela de agregação.

---

# 12. Etapa 2.3 — Sweep heterogêneo

## 12.1. Objetivo

Identificar a nova região de transição considerando:

```text
eMBB + URLLC + mMTC
```

## 12.2. Pontos iniciais

Com base na Rodada 1b, começar por:

\[
\lambda \in \{2,4,5,6,7\}.
\]

Manter \(\lambda=8\) somente como controle de estresse severo, caso necessário.

---

# 13. Atenção: \(\lambda\) sozinho não será mais suficiente

Com slices heterogêneos, um UE eMBB e um dispositivo mMTC não representam a mesma pressão de carga.

Registrar também:

\[
N_{eMBB},
N_{URLLC},
N_{mMTC}
\]

e:

\[
L_{eMBB},
L_{URLLC},
L_{mMTC}.
\]

Criar uma medida agregada de carga oferecida:

\[
L_{total}
=
L_{eMBB}
+
L_{URLLC}
+
L_{mMTC}.
\]

Mas manter as componentes separadas no CSV.

---

# 14. Composição de tráfego

Além do sweep global, testar pelo menos algumas composições:

### Perfil A — balanceado

Carga distribuída entre os três slices.

### Perfil B — eMBB dominante

Burst ou aumento de tráfego eMBB.

### Perfil C — mMTC dominante

Grande quantidade de dispositivos ativos.

### Perfil D — URLLC burst

Aumento temporário de tráfego sensível a latência.

### Perfil E — combinado

eMBB + mMTC aumentam enquanto URLLC mantém requisito crítico.

---

# 15. Métricas do sweep

Por janela:

- número de UEs por slice;
- carga oferecida por slice;
- recursos solicitados;
- recursos programados;
- recursos executados;
- recursos ociosos;
- throughput eMBB;
- latency URLLC;
- PDR URLLC;
- PDR mMTC;
- fila por slice;
- SLA OK/FAIL por slice;
- severidade da violação;
- SNR;
- CQI;
- RLF;
- utilização DL ativa;
- utilização total da capacidade DL.

---

# 16. Duas métricas de utilização

Manter explicitamente:

## Utilização condicionada à demanda

\[
U_{\text{active-DL}}
=
\frac{
PRBs_{\text{executed}}
}{
C \times DL_{\text{slots com demanda}}
}.
\]

## Utilização global DL

\[
U_{\text{all-DL}}
=
\frac{
PRBs_{\text{executed}}
}{
C \times DL_{\text{slots totais}}
}.
\]

As duas devem ser reportadas separadamente.

---

# 17. Etapa 2.4 — Oracle de viabilidade

Esta é a etapa central da Rodada 2.

## Objetivo

Para cada estado/janela \(t\), responder:

> **Existe alguma alocação de recursos capaz de satisfazer simultaneamente todos os SLAs?**

---

# 18. Definição formal

Para três slices:

\[
\mathbf{a}_t
=
(a_t^{eMBB},a_t^{URLLC},a_t^{mMTC})
\]

sujeito a:

\[
a_t^{eMBB}
+
a_t^{URLLC}
+
a_t^{mMTC}
\le C.
\]

Definir:

\[
\mathcal{F}_t
=
\left\{
\mathbf{a}_t:
SLA_i(\mathbf{a}_t,z_t)=1,\ \forall i
\right\}.
\]

Se:

\[
\mathcal{F}_t\neq\emptyset,
\]

o estado é:

```text
JOINTLY FEASIBLE
```

Se:

\[
\mathcal{F}_t=\emptyset,
\]

o estado é:

```text
JOINTLY INFEASIBLE
```

---

# 19. Violação evitável

Se ocorreu uma violação:

\[
V_t=1
\]

mas:

\[
\mathcal{F}_t\neq\emptyset,
\]

classificar como:

\[
\boxed{\text{AVOIDABLE}}
\]

ou seja:

> havia capacidade e pelo menos uma ação válida capaz de preservar todos os SLAs.

---

# 20. Violação em estado inviável

Se:

\[
V_t=1
\]

e:

\[
\mathcal{F}_t=\emptyset,
\]

classificar como:

\[
\boxed{\text{UNAVOIDABLE AT SYSTEM LEVEL}}
\]

Significado:

> não era possível preservar simultaneamente todos os SLAs.

---

# 21. Cuidado conceitual

Quando:

\[
\mathcal{F}_t=\emptyset,
\]

isso não significa que a violação de **cada slice individual** era inevitável.

Pode existir:

```text
ação A → salva URLLC, viola eMBB
ação B → salva eMBB, viola mMTC
ação C → salva mMTC, viola eMBB
```

Logo, registrar também:

- número mínimo de slices que precisam violar;
- qual combinação produz menor déficit;
- quais SLAs ainda podem ser preservados;
- severidade mínima possível.

---

# 22. Oracle — implementação preferida

## Busca exaustiva

Se o espaço de ações permitir, enumerar todas as alocações:

```text
for a_embb in actions:
    for a_urllc in actions:
        for a_mmtc in actions:

            if a_embb + a_urllc + a_mmtc <= C:

                evaluate()

                if all_slas_ok:
                    feasible = True
```

Registrar:

```text
num_feasible_actions
best_feasible_action
min_total_deficit_action
```

---

# 23. Oracle — requisito de reprodutibilidade

Comparar diferentes ações somente sob condições equivalentes.

Garantir:

- mesma seed;
- mesmos UEs;
- mesmos arrivals;
- mesmos tempos de sessão;
- mesmos traces;
- mesma posição;
- mesmo tráfego;
- mesmo canal.

Idealmente:

```text
snapshot state
    ↓
executar ação A
    ↓
restore snapshot
    ↓
executar ação B
```

Se o simulador não permitir snapshot/restore:

- usar episódios curtos reproduzíveis;
- usar traces determinísticos;
- ou construir um modelo offline validado para o oracle.

Documentar claramente a estratégia.

---

# 24. Métricas do oracle

Calcular:

## Probabilidade de inviabilidade

\[
P_{\text{inf}}
=
P(\mathcal{F}_t=\emptyset).
\]

## Probabilidade de viabilidade

\[
P_{\text{feas}}
=
1-P_{\text{inf}}.
\]

## Avoidable Violation Rate

\[
AVR
=
P(
V_t=1
\mid
\mathcal{F}_t\neq\emptyset
).
\]

## Unavoidable-state Violation Rate

\[
UVR
=
P(
V_t=1
\mid
\mathcal{F}_t=\emptyset
).
\]

## Tamanho do conjunto factível

\[
|\mathcal{F}_t|.
\]

## Severidade mínima possível

\[
D_{\min}(t)
=
\min_{\mathbf a}
\sum_i w_iD_i(\mathbf a).
\]

---

# 25. Classificação da fronteira

Depois do oracle, classificar empiricamente:

## Normal

\[
P_{\text{inf}}\approx0
\]

e:

\[
|\mathcal{F}_t|
\]

alto.

## Marginal

\[
P_{\text{inf}}\approx0
\]

mas:

\[
|\mathcal{F}_t|
\]

pequeno.

## Overload transitório

\[
0<P_{\text{inf}}<1.
\]

Essa deve ser a região principal do novo artigo.

## Overload severo

\[
P_{\text{inf}}\rightarrow1.
\]

Usar apenas como teste de limite.

---

# 26. Gate científico da Rodada 2

Ao final, produzir uma tabela como:

| Carga | Violação observada | \(P_{\text{inf}}\) | AVR | \(|\mathcal F|\) médio | Severidade mínima |
|---|---:|---:|---:|---:|---:|
| \(\lambda2\) | ... | ... | ... | ... | ... |
| \(\lambda4\) | ... | ... | ... | ... | ... |
| \(\lambda5\) | ... | ... | ... | ... | ... |
| \(\lambda6\) | ... | ... | ... | ... | ... |
| \(\lambda7\) | ... | ... | ... | ... | ... |

Essa tabela deve decidir a arquitetura final da proposta.

---

# 27. Etapa 2.5 — Episódio transitório

Somente depois do oracle.

Escolher:

\[
L_N
\]

como regime normal.

Escolher:

\[
L_O
\]

como overload moderado identificado empiricamente.

Não fixar antecipadamente:

\[
L_O=\lambda7.
\]

O oracle pode mostrar que \(\lambda6\) representa melhor a região transitória.

---

# 28. Estrutura do episódio

Exemplo:

```text
0–200 s     NORMAL
200–400 s   OVERLOAD
400–600 s   NORMAL
```

Ou equivalente após ajuste de warm-up.

Objetivo:

```text
Viável
  ↓
Marginal
  ↓
Inviável em parte das janelas
  ↓
Marginal
  ↓
Viável
```

---

# 29. O que medir na transição

Por janela:

- carga por slice;
- estado viável/inviável;
- número de ações factíveis;
- SLA por slice;
- severidade;
- utilização;
- fila;
- latência;
- PDR;
- throughput;
- tempo de entrada em overload;
- tempo de recuperação;
- comportamento do controlador.

---

# 30. Decisão sobre o SQR-JC

Depois da Rodada 2:

## Caso A — muitas violações evitáveis

Se:

\[
AVR
\]

for relevante:

priorizar:

```text
Joint Coordination
+
preservação dos mínimos seguros
```

## Caso B — muitos estados inviáveis

Se:

\[
P_{\text{inf}}
\]

for relevante:

priorizar:

```text
Graceful Degradation
+
anti-windup
```

## Caso C — ambos aparecem

Implementar SQR-JC completo:

```text
Feasibility Detection
+
Joint Coordination
+
Graceful Degradation
+
Anti-windup
```

---

# 31. Hipótese a testar posteriormente

A hipótese candidata será:

> **Um mecanismo de coordenação inter-slice consciente da viabilidade consegue reduzir violações evitáveis em estados conjuntamente viáveis e reduzir a severidade das violações inevitáveis em períodos de sobrecarga transitória, sem comprometer a eficiência de recursos em condições normais.**

---

# 32. CSV principal recomendado

```csv
seed,
time,
phase,
lambda,
num_ue_embb,
num_ue_urllc,
num_ue_mmtc,
offered_embb_mbps,
offered_urllc_mbps,
offered_mmtc_mbps,
allocated_embb,
allocated_urllc,
allocated_mmtc,
scheduled_embb,
scheduled_urllc,
scheduled_mmtc,
executed_embb,
executed_urllc,
executed_mmtc,
prb_idle,
util_active_dl,
util_all_dl,
throughput_embb,
latency_urllc_mean,
latency_urllc_p95,
latency_urllc_p99,
pdr_urllc,
pdr_mmtc,
sla_embb_ok,
sla_urllc_ok,
sla_mmtc_ok,
deficit_embb,
deficit_urllc,
deficit_mmtc,
joint_feasible,
num_feasible_actions,
avoidable_violation,
unavoidable_state,
min_total_deficit,
sinr_p05,
sinr_mean,
cqi_p05,
cqi_mean,
rlf_count
```

---

# 33. Artefatos esperados

```text
results/sqr_characterization_round2/
├── raw/
├── processed/
├── heterogeneous/
├── oracle/
├── transient/
├── plots/
└── reports/
```

Arquivos principais:

```text
round2_stability.csv
round2_heterogeneous_sweep.csv
round2_feasibility_windows.csv
round2_oracle_summary.csv
round2_avoidable_vs_unavoidable.csv
round2_transient.csv
```

---

# 34. Figuras obrigatórias

## Figura 1 — carga × utilização

Mostrar:

- \(U_{\text{active-DL}}\);
- \(U_{\text{all-DL}}\).

## Figura 2 — carga × SLA violation

Uma curva por slice.

## Figura 3 — carga × \(P_{\text{inf}}\)

\[
P(\mathcal{F}_t=\emptyset).
\]

## Figura 4 — carga × \(|\mathcal F_t|\)

Mostrar desaparecimento progressivo das ações factíveis.

## Figura 5 — violações evitáveis × estados inviáveis

Separar:

```text
avoidable
unavoidable-state
```

## Figura 6 — qualidade de rádio

Mostrar:

- SNR;
- CQI;
- RLF;

para provar que a fronteira não foi criada destruindo o rádio.

## Figura 7 — episódio transitório

Mostrar no eixo temporal:

- carga;
- viabilidade;
- SLAs;
- utilização;
- déficit.

---

# 35. Checklist da Rodada 2

## Estabilidade

- [ ] \(\lambda2\) reexecutado com warm-up.
- [ ] \(\lambda5\) reexecutado com warm-up.
- [ ] \(\lambda7\) reexecutado com warm-up.
- [ ] Métricas estabilizadas após warm-up.

## Heterogeneidade

- [ ] eMBB implementado.
- [ ] URLLC implementado.
- [ ] mMTC implementado.
- [ ] Latência instrumentada.
- [ ] PDR instrumentado.
- [ ] Deadline miss instrumentado.
- [ ] SLAs definidos com fonte.

## Validação

- [ ] Baixa carga atende eMBB.
- [ ] Baixa carga atende URLLC.
- [ ] Baixa carga atende mMTC.
- [ ] RLF não domina os resultados.

## Sweep

- [ ] \(\lambda2\) executado.
- [ ] \(\lambda4\) executado.
- [ ] \(\lambda5\) executado.
- [ ] \(\lambda6\) executado.
- [ ] \(\lambda7\) executado.
- [ ] Múltiplas seeds.

## Oracle

- [ ] Oracle implementado.
- [ ] Reprodutibilidade validada.
- [ ] \(\mathcal F_t\) calculado.
- [ ] \(|\mathcal F_t|\) registrado.
- [ ] AVR calculado.
- [ ] Estados inviáveis classificados.
- [ ] Severidade mínima calculada.

## Transição

- [ ] \(L_N\) definido pelos dados.
- [ ] \(L_O\) definido pelos dados.
- [ ] Episódio Normal → Overload → Normal executado.
- [ ] Recuperação medida.

---

# 36. Critério de encerramento da Rodada 2

A Rodada 2 será considerada concluída quando for possível responder quantitativamente:

1. **Em quais estados todos os SLAs ainda podem ser satisfeitos?**
2. **Em quais estados isso deixa de ser possível?**
3. **Qual é a região de transição?**
4. **Quantas violações observadas eram evitáveis?**
5. **Quantas ocorreram em estados conjuntamente inviáveis?**
6. **Qual a severidade mínima possível nesses estados?**
7. **Qual carga deve ser usada como \(L_N\)?**
8. **Qual carga deve ser usada como \(L_O\)?**
9. **Qual componente do SQR-JC é realmente necessário?**

---

# 37. Saída esperada para a orientação

Ao final, devemos conseguir apresentar algo como:

> "Primeiro demonstramos que a degradação pode surgir exclusivamente pelo aumento da demanda, mantendo a camada de rádio saudável. Em seguida, introduzimos slices heterogêneos e construímos um oracle de viabilidade para avaliar, em cada janela, se existia alguma alocação capaz de satisfazer simultaneamente os SLAs. Isso nos permitiu separar violações evitáveis de estados em que alguma degradação era inevitável. A partir dessa fronteira, definimos o cenário de sobrecarga transitória e identificamos exatamente qual mecanismo de coordenação precisa ser introduzido no SQR."

---

# 38. Regra principal desta rodada

Não implementar o SQR-JC antes de responder:

\[
\boxed{
\text{quanto do problema é evitável por decisão}
}
\]

e:

\[
\boxed{
\text{quanto decorre de inviabilidade física temporária}
}
\]

A Rodada 2 existe justamente para produzir essa evidência.
