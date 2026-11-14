# ETAPA 2.5 — EMENDA G4-A
## Calibração qualificada dos quantis auxiliares q25/q75
### SQR-JC — Safe Quantile Regression with Feasibility-Aware Joint Coordination

**Status:** EMENDA METODOLÓGICA PRÉ-PILOT  
**Aplicação:** substituir apenas a decisão do G4 no protocolo final da Etapa 2.5.  
**Não substitui:** os demais gates, regras, seeds, cenário, baselines ou critérios do protocolo final.

---

## 1. Contexto

Durante o G4 foi identificado um bug de implementação nas heads auxiliares `q25/q75`.

As heads estavam sendo atualizadas comparando o outcome `y` com:

```text
f_safe_before
```

em vez de utilizar suas próprias previsões:

```text
f_q25_before
f_q75_before
```

Como `f_safe` utiliza `tau ≈ 0.01–0.05` e é pessimista por construção, isso gerava atualização positiva crônica nas heads auxiliares e cobertura degenerada.

O bug foi corrigido:

```text
q25 update -> compare y with f_q25_before
q75 update -> compare y with f_q75_before
```

O CAL foi reprocessado offline com os mesmos dados.

Classificação:

```text
BUG FIX
```

e não:

```text
retuning pós-hoc
```

---

## 2. Resultado pós-fix

Com `lengthscale = 1.0`, as coberturas ficaram próximas dos alvos:

```text
eMBB:
    q25 ≈ 0.18
    q75 ≈ 0.70

URLLC:
    q25 ≈ 0.23
    q75 ≈ 0.75

mMTC:
    q25 ≈ 0.25
    q75 ≈ 0.75
```

O critério nominal:

```text
|coverage - tau| <= 0.10
```

foi satisfeito.

O único FAIL restante veio do critério:

```text
tau alvo dentro do 95% block-bootstrap CI
```

---

## 3. Sweep autorizado de lengthscale

Foi executado apenas o grid autorizado:

```text
original_lengthscale × {0.5, 1.0, 2.0}
```

Resultados:

| lengthscale | eMBB q25/q75 | URLLC q25/q75 | mMTC q25/q75 | pior erro absoluto |
|---|---:|---:|---:|---:|
| 0.5 | 0.16 / 0.62 | 0.23 / 0.73 | 0.25 / 0.75 | ~0.127 |
| 1.0 | 0.18 / 0.68 | 0.23 / 0.74 | 0.25 / 0.75 | ~0.074 |
| **2.0** | **0.19 / 0.72** | **0.23 / 0.75** | **0.24 / 0.74** | **~0.063** |

Conclusão:

```text
lengthscale = 2.0
```

é a melhor configuração dentro do espaço de calibração autorizado.

---

## 4. Interpretação correta do FAIL original

O G4 original NÃO deve ser marcado retroativamente como PASS.

Manter:

```text
G4 original = FAIL / STOP
```

Motivo:

```text
o alvo nominal tau não ficou dentro do CI bootstrap em todas as heads
```

Interpretar como:

> existe um pequeno viés sistemático residual nos quantis auxiliares.

Não escrever:

```text
"o CI está errado"
"o CI é inválido"
"não existe viés"
```

---

## 5. Emenda G4-A

Criar novo gate:

```text
G4-A — Calibration-Qualified Feasibility Gate
```

O G4-A permite seguir para o próximo estágio SE a calibração residual não produzir falsas classificações confidentes em taxa inaceitável.

---

## 6. Configuração congelada

Usar:

```text
lengthscale = 2.0
support threshold = 90
```

A partir deste ponto:

```text
FREEZE
```

Não alterar `lengthscale` ou `threshold` usando PILOT ou MAIN.

Registrar:

```text
kernel_lengthscale: 2.0
sigma_support_threshold: 90
decision_source: G4-A amendment
```

---

## 7. Condições mínimas já satisfeitas

Para `ls=2.0`:

```text
max absolute coverage error <= 0.063
```

Logo:

```text
criterion_abs_error <= 0.10 -> PASS
```

O `thr=90` fica congelado como configuração operacional conservadora.

---

## 8. Nuance de segurança dos quantis

Não assumir que todo o viés residual é automaticamente conservador.

### q25

Se:

```text
coverage(q25) < 0.25
```

o q25 tende a ser mais pessimista.

Como FEASIBLE exige:

```text
q25_margin >= 0
```

isso tende a:

```text
reduzir declarações FEASIBLE
aumentar MARGINAL
```

### q75

Se:

```text
coverage(q75) < 0.75
```

o q75 pode ficar ligeiramente subestimado.

Como INFEASIBLE usa:

```text
q75_margin < 0
```

isso pode tornar a declaração INFEASIBLE um pouco mais fácil.

Logo é obrigatório validar empiricamente:

```text
False Feasible Rate
False Infeasible Rate
```

---

## 9. Novo gate de falsa classificação

Antes de liberar o PILOT, executar validação independente usando somente CAL held-out / replay permitido pelo protocolo.

Medir nas classes confidentes:

```text
FFR = False Feasible Rate
FIR = False Infeasible Rate
```

### False Feasible

Estado/ação classificado como:

```text
FEASIBLE
```

mas cujo outcome observado/replay mostra:

```text
SLA violation
```

Definição:

```text
FFR =
#(FEASIBLE predicted and violated)
/
#(FEASIBLE predicted)
```

### False Infeasible

Estado/ação classificado como:

```text
INFEASIBLE
```

mas cujo outcome observado/replay mostra:

```text
all-SLA-OK
```

Definição:

```text
FIR =
#(INFEASIBLE predicted and all-SLA-OK)
/
#(INFEASIBLE predicted)
```

---

## 10. Tratamento de denominador zero

Se não houver nenhuma predição FEASIBLE:

```text
FFR = NA
```

Se não houver nenhuma predição INFEASIBLE:

```text
FIR = NA
```

Não substituir `NA` por zero.

Nesse caso, o gate não pode ser considerado automaticamente PASS.

---

## 11. Critério G4-A

PASS se:

```text
FFR <= 0.10
FIR <= 0.10
```

nas classes confidentes.

Não incluir no cálculo:

```text
MARGINAL
UNKNOWN
OUT_OF_SUPPORT
```

---

## 12. Validação recomendada

Usar estados CAL que NÃO foram usados para escolher o `sigma_support_threshold`.

Estrutura recomendada:

```text
CAL-fit
CAL-validation
```

Se o CAL já foi todo usado, criar split reproduzível por seed ou bloco temporal.

Preferência:

```text
seed 1001 -> fit
seed 1002 -> fit
seed 1003 -> validation
```

ou equivalente compatível com os dados já existentes.

É permitido rerodar CAL.

Não usar:

```text
PILOT
MAIN
```

para aprovar o G4-A.

---

## 13. Tabela obrigatória

Gerar:

| Predicted class | Observed/replay result | Count | Rate |
|---|---|---:|---:|
| FEASIBLE | all-SLA-OK | n | — |
| FEASIBLE | violated | n | FFR |
| INFEASIBLE | violated | n | — |
| INFEASIBLE | all-SLA-OK | n | FIR |
| MARGINAL | any | n | descriptive only |
| UNKNOWN/OOS | any | n | descriptive only |

Apresentar também por slice, quando aplicável:

```text
eMBB
URLLC
mMTC
joint classification
```

O gate principal é a classificação conjunta.

---

## 14. Replay

Quando possível, preferir:

```text
deterministic replay
```

para validar um `INFEASIBLE`.

Para `False Infeasible`, a evidência forte é:

```text
mesmo estado
ação alternativa válida
all-SLA-OK
```

Não concluir inviabilidade apenas porque a ação executada violou.

---

## 15. Linguagem obrigatória

Usar:

```text
predicted FEASIBLE
predicted INFEASIBLE
confirmed avoidable
not recovered within tested action space
```

Evitar:

```text
physically impossible
mathematically infeasible
guaranteed feasible
```

sem prova correspondente.

---

## 16. Decisão após G4-A

### Caso A — PASS

Se:

```text
FFR <= 0.10
FIR <= 0.10
```

então:

```text
G4-A = PASS
lengthscale = 2.0 FREEZE
threshold = 90 FREEZE
prosseguir para G5
```

### Caso B — FFR > 0.10

```text
STOP
revisar feature/kernel/calibração
não iniciar PILOT
```

### Caso C — FIR > 0.10

```text
STOP
revisar feature/kernel/calibração
não iniciar PILOT
```

### Caso D — classes confidentes insuficientes

```text
STOP
reportar insufficient confident-class support
```

Não ajustar threshold usando PILOT.

---

## 17. G5 continua inalterado

O benchmark exaustivo com `ls=2.0` apresentou aproximadamente:

```text
p95 ≈ 14.7 s
```

Logo:

```text
exhaustive runtime search = FAIL para execução online
```

Aplicar o protocolo original:

```text
coarse-to-fine
```

Critérios:

```text
feasibility-class agreement >= 98%
median normalized objective regret <= 2%
p95 normalized objective regret <= 5%
p95 decision time <= 80 ms
```

---

## 18. Proibição de tuning adicional

Após registrar:

```text
ls=2.0
thr=90
```

é proibido testar, para melhorar G4-A:

```text
novo lengthscale
novo threshold
novo eta
novo tau
novas features
novo kernel
```

Se G4-A falhar:

```text
STOP metodológico
```

e abrir nova versão de protocolo.

---

## 19. Atualização do manifest

Adicionar:

```yaml
g4_original:
  status: FAIL
  reason: target_tau_outside_block_bootstrap_ci

g4_amendment:
  id: G4-A
  status: PENDING
  rationale: approximate quantile calibration accepted conditionally
  max_abs_coverage_error: 0.063

kernel:
  family: Matern
  lengthscale: 2.0
  frozen: true

support:
  threshold_percentile: 90
  frozen: true

g4a_thresholds:
  max_false_feasible_rate: 0.10
  max_false_infeasible_rate: 0.10
```

---

## 20. Relatório obrigatório

Gerar:

```text
docs/artigo/G4A_validation_report.md
```

contendo:

1. descrição do bug original;
2. commit da correção;
3. confirmação da suíte de testes;
4. cobertura pré-fix;
5. cobertura pós-fix;
6. sweep `0.5/1.0/2.0`;
7. justificativa para `ls=2.0`;
8. justificativa para `thr=90`;
9. block-bootstrap results;
10. reconhecimento explícito do viés residual;
11. split de validação;
12. tabela FFR/FIR;
13. contagem de FEASIBLE/MARGINAL/INFEASIBLE;
14. suporte/OOS;
15. decisão `PASS/STOP`;
16. próximos passos.

---

## 21. Texto recomendado para o relatório científico

Se G4-A passar:

> **The auxiliary q25/q75 heads exhibited a small but systematic residual calibration bias after correction of an implementation bug. Within the pre-specified kernel length-scale grid, the best configuration (length-scale 2.0) achieved a maximum absolute empirical coverage error of approximately 0.063 across slices and quantiles. Because the auxiliary heads are used for conservative feasibility classification rather than probabilistic inference, we accepted this approximate calibration only after independently verifying that false-feasible and false-infeasible rates remained below 0.10 in held-out calibration states.**

Não afirmar:

```text
perfect calibration
exact 25/75 quantiles
CI passed
```

---

## 22. Comando ao agente

```text
Aplicar a Emenda G4-A antes de prosseguir para o PILOT.

1. Manter G4 original como FAIL formal.
2. Congelar lengthscale = 2.0.
3. Congelar support threshold = 90.
4. Não alterar eta, tau, features, kernel family ou SLA.
5. Construir validação CAL held-out reproduzível.
6. Calcular classificação conjunta:
   FEASIBLE / MARGINAL / INFEASIBLE.
7. Calcular FFR e FIR apenas nas classes confidentes.
8. Preferir replay determinístico para confirmar falsos INFEASIBLE quando aplicável.
9. Gerar tabela de contingência completa.
10. Gerar docs/artigo/G4A_validation_report.md.

Gate:

PASS somente se:
    FFR <= 0.10
    FIR <= 0.10

Se PASS:
    G4-A = PASS
    manter ls=2.0/thr=90 FREEZE
    avançar para G5.

Se FAIL:
    STOP
    não iniciar PILOT
    não fazer tuning adicional fora do protocolo.
```

---

## 23. Decisão científica incorporada

A decisão não é:

> “ignorar que o CI falhou”.

A decisão é:

> **reconhecer o pequeno viés de calibração, manter a transparência sobre o FAIL do critério original e substituir a exigência de calibração exata por uma validação operacional independente daquilo que realmente importa para o SQR-JC: não produzir falsas classificações confidentes em taxa inaceitável.**

---

## 24. Estado após esta emenda

```text
G4 original = FAIL / documented
G4-A = PENDING validation
ls = 2.0 = FROZEN
thr = 90 = FROZEN
PILOT = BLOCKED until G4-A PASS
```
