# ETAPA 2.5 — G4-B
## Última recalibração controlada antes do PILOT
### SQR-JC — Safe Quantile Regression with Feasibility-Aware Joint Coordination

**Status:** EMENDA METODOLÓGICA PRÉ-PILOT  
**Objetivo:** tratar de forma controlada os dois mecanismos de erro identificados no G4-A, sem abrir novo tuning genérico do learner.  
**Regra:** o PILOT permanece bloqueado até a conclusão desta etapa.

---

## 1. Contexto

O G4-A terminou em **STOP**.

Foram identificados dois problemas distintos:

1. **FFR da URLLC**, concentrado em estados de carga alta e associado à instabilidade da taxa de miss em janelas de 100 ms;
2. **FIR do eMBB**, concentrado em `lambda = 5/6`, associado à subestimação do `q75` em alta carga.

A G4-B deve tratar somente esses dois pontos.

Não realizar tuning indiscriminado do SQR-JC.

---

## 2. Congelar tudo que NÃO está autorizado a mudar

Manter exatamente:

```text
kernel = Matérn
lengthscale = 2.0
support threshold = 90

eta = atual
tau_safe = atual
q25 = 0.25
q75 = 0.75

feature set = atual
feature normalization = atual

eMBB SLA threshold = atual
URLLC packet deadline = 10 ms
URLLC miss threshold = 5%
mMTC SLA/PDR = atual

radio scenario = atual
scheduler = atual
TDD = atual
decision period = 100 ms
resource semantics = atual
```

Não testar:

```text
novo kernel
novo lengthscale
novo threshold
novo eta
novo tau
novas features
novo SLA eMBB
novo deadline URLLC
```

---

## 3. Alteração autorizada A — URLLC

Manter:

```text
deadline por pacote = 10 ms
miss threshold = 5%
decision period = 100 ms
```

Alterar SOMENTE a janela usada para calcular a taxa de miss utilizada pelo modelo/classificador.

Substituir:

```text
miss rate calculado apenas na janela de 100 ms
```

por:

```text
rolling miss rate de 1 segundo
```

atualizada a cada decisão de 100 ms.

Ou seja:

```text
decision every 100 ms
URLLC risk observed over previous 1 s
```

Objetivo:

```text
reduzir a variância da cauda
sem alterar o deadline de 10 ms
```

---

## 4. Atualização do SLA manifest da URLLC

Registrar explicitamente:

```text
URLLC packet deadline: 10 ms
URLLC miss threshold: 5%
URLLC aggregation window: rolling 1 s
controller decision period: 100 ms
```

Não apresentar isso como:

```text
deadline URLLC = 1 s
```

O deadline continua:

```text
10 ms por pacote
```

A mudança é apenas na agregação temporal usada para estimar a taxa de miss.

---

## 5. Tratamento do início da simulação para rolling 1 s

Antes de existir histórico completo de 1 s:

```text
t < 1 s
```

usar:

```text
warmup / insufficient-history flag
```

Essas janelas NÃO devem ser usadas para:

```text
calibration
FFR
FIR
gate decision
```

A partir de histórico completo:

```text
rolling window = últimos 1000 ms
```

---

## 6. Alteração autorizada B — eMBB

Não alterar o modelo do eMBB.

Adicionar somente mais dados CAL de alta carga para o conjunto de ajuste.

Cargas:

```text
lambda = 5
lambda = 6
```

Objetivo:

```text
aumentar a cobertura de estados de alta carga
sem mudar learner
```

---

## 7. CAL extra de alta carga

Executar:

```text
lambda = 5 -> 3 novas seeds
lambda = 6 -> 3 novas seeds
```

Criar conjunto:

```text
CAL_EXTRA_FIT
```

Essas seeds NÃO podem pertencer a:

```text
PILOT
MAIN
validation final
```

Usar somente para fit/calibração.

---

## 8. Validação independente obrigatória

Manter um conjunto de validação que não seja usado:

```text
nem para fit
nem para escolher threshold
nem para escolher hyperparameters
```

Como:

```text
lengthscale = 2.0
threshold = 90
```

já estão congelados, a validação serve apenas para verificar o resultado da G4-B.

Se necessário, gerar novas seeds exclusivamente para:

```text
G4B_VALIDATION
```

com:

```text
lambda = {2,5,6}
```

Nenhuma dessas seeds entra no treinamento.

---

## 9. Não usar PILOT ou MAIN

É proibido utilizar:

```text
PILOT seeds
MAIN seeds
```

para aprovar a G4-B.

O PILOT permanece bloqueado.

---

## 10. Recalibração após as mudanças autorizadas

Treinar novamente os modelos auxiliares usando:

```text
CAL original válido
+
CAL_EXTRA_FIT de lambda 5/6
```

Manter inalterados:

```text
lengthscale = 2.0
threshold = 90
q25 = 0.25
q75 = 0.75
eta
features
normalization
kernel family
```

---

## 11. Avaliações obrigatórias

Na validação independente calcular:

### Quantile coverage

Por slice:

```text
q25 empirical coverage
q75 empirical coverage
absolute coverage error
```

### Classificação conjunta

```text
FEASIBLE
MARGINAL
INFEASIBLE
UNKNOWN/OOS
```

### Falsas classificações

```text
FFR = False Feasible Rate
FIR = False Infeasible Rate
```

---

## 12. Gate principal da G4-B

PASS somente se:

```text
FFR <= 0.10
FIR <= 0.10
```

na classificação conjunta.

Também verificar:

```text
FEASIBLE all-OK rate
MARGINAL all-OK rate
INFEASIBLE all-OK rate
```

Esperado:

```text
P(all-OK | FEASIBLE)
>
P(all-OK | MARGINAL)
>
P(all-OK | INFEASIBLE)
```

---

## 13. Diagnóstico específico — URLLC

Comparar:

```text
FFR com janela 100 ms
vs
FFR com rolling 1 s
```

Verificar especificamente:

```text
lambda = 5
lambda = 6
```

Reportar:

```text
false-FEASIBLE total
false-FEASIBLE causados por URLLC
miss rate distribution
p95/p99 latency
quantile coverage
```

Objetivo:

```text
confirmar se a agregação temporal reduz o problema
sem alterar o deadline
```

---

## 14. Diagnóstico específico — eMBB

Comparar:

```text
FIR anterior
vs
FIR após CAL_EXTRA_FIT
```

especialmente em:

```text
lambda = 5
lambda = 6
```

Reportar:

```text
q75 coverage
margin distribution
false-INFEASIBLE count
throughput observed
support density
```

Objetivo:

```text
verificar se o FIR era causado por baixa cobertura da região de alta carga
```

---

## 15. O que caracteriza sucesso

A G4-B NÃO precisa produzir:

```text
quantile coverage perfeita
```

Ela precisa demonstrar que o detector é operacionalmente confiável para o objetivo do SQR-JC:

```text
FFR <= 10%
FIR <= 10%
```

sem novo tuning do learner.

---

## 16. Se G4-B PASSAR

Se:

```text
FFR <= 0.10
FIR <= 0.10
```

então:

```text
G4-B = PASS
```

Congelar definitivamente:

```text
URLLC rolling window = 1 s
lengthscale = 2.0
threshold = 90
CAL configuration
all model settings
```

Depois:

```text
prosseguir para G5
```

E somente após:

```text
G5 = PASS
```

liberar:

```text
PILOT
```

---

## 17. Se G4-B FALHAR

Se:

```text
FFR > 0.10
ou
FIR > 0.10
```

então:

```text
G4-B = STOP FINAL
```

Não criar automaticamente:

```text
G4-C
```

Não testar automaticamente:

```text
novas features
novos thresholds
novo eta
novos quantis
novos kernels
```

Gerar diagnóstico final e levar a decisão ao orientador.

---

## 18. Relatório obrigatório

Gerar:

```text
docs/artigo/G4B_validation_report.md
```

contendo:

1. referência ao FAIL do G4-A;
2. configuração congelada;
3. definição exata do rolling 1 s da URLLC;
4. atualização do `sla_manifest`;
5. novas seeds CAL de `lambda=5/6`;
6. seeds exclusivas de validação;
7. coverage q25/q75;
8. FFR;
9. FIR;
10. tabela FEASIBLE/MARGINAL/INFEASIBLE;
11. diagnóstico separado URLLC/eMBB;
12. comparação G4-A vs G4-B;
13. decisão PASS/STOP;
14. indicação explícita se G5/PILOT estão liberados ou bloqueados.

---

## 19. Atualização de manifest

Adicionar algo equivalente a:

```yaml
g4b:
  status: PENDING

kernel:
  family: Matern
  lengthscale: 2.0
  frozen: true

support:
  threshold_percentile: 90
  frozen: true

urllc:
  packet_deadline_ms: 10
  miss_threshold: 0.05
  aggregation_window_ms: 1000
  decision_period_ms: 100

embb:
  calibration_extra_loads:
    - 5
    - 6
  extra_fit_seeds_per_load: 3

g4b_gate:
  max_false_feasible_rate: 0.10
  max_false_infeasible_rate: 0.10
```

---

## 20. Comando direto ao agente

```text
Executar G4-B como última recalibração controlada antes do PILOT.

1. Manter kernel Matérn, ls=2.0 e thr=90 congelados.
2. Não alterar eta, tau, features, normalização ou SLAs.
3. Para URLLC:
   - manter deadline de 10 ms;
   - manter miss threshold de 5%;
   - substituir apenas a agregação de 100 ms por rolling 1 s;
   - atualizar a cada 100 ms;
   - excluir warmup <1 s da validação.

4. Para eMBB:
   - adicionar 3 novas CAL-fit seeds em lambda=5;
   - adicionar 3 novas CAL-fit seeds em lambda=6;
   - não alterar learner.

5. Manter validation independente.
6. Não usar PILOT ou MAIN para G4-B.
7. Recalcular q25/q75 coverage.
8. Recalcular classificação conjunta.
9. Calcular FFR e FIR.
10. Gerar docs/artigo/G4B_validation_report.md.

PASS somente se:
    FFR <= 0.10
    FIR <= 0.10

Se PASS:
    G4-B = PASS
    prosseguir para G5.

Se FAIL:
    G4-B = STOP FINAL
    não criar G4-C automaticamente
    levar decisão ao orientador.
```

---

## 21. Regra final

Esta G4-B é a:

```text
última tentativa metodológica pré-registrada antes do PILOT
```

Fluxo:

```text
G4-B PASS
    ↓
G5
    ↓
G5 PASS
    ↓
PILOT
```

ou:

```text
G4-B FAIL
    ↓
STOP FINAL
    ↓
orientador
```
