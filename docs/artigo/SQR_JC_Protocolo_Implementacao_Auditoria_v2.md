# PROTOCOLO DE IMPLEMENTAÇÃO, AUDITORIA E DECISÃO — SQR-HET × JOINT SELECTION

**Projeto:** RAN slicing em O-RAN — SQR-HET versus seleção conjunta de ações  
**Versão:** 2.0 — 07/11/2026 (inclui auditoria crítica prévia dos replays e das figuras)  
**Destinatário:** agente implementador/orquestrador de simulações (Python + ns-3/5G-LENA)  
**Natureza:** protocolo operacional e critérios de auditoria para um agente executor, **não** relatório de resultados nem comprovação já realizada  
**Estado:** `D0-A AUTORIZADO PRIMEIRO; D0-B/D1–D3 CONDICIONADOS AO GATE DE INTEGRIDADE; D4 PILOT/D5 MAIN BLOQUEADOS`.

> **MISSÃO DO AGENTE:** primeiro auditar a integridade das evidências existentes (especialmente os cinco replays, R19 e unidade de recursos); somente depois determinar se há evidência experimental de que a **seleção independente por slice, seguida de projeção de capacidade**, perde oportunidades de atendimento simultâneo dos SLAs que poderiam ser aproveitadas por uma **seleção conjunta usando os mesmos modelos SQR**. **Não implementar um SQR-JC completo antes dessa demonstração.** Não inferir sucesso ou criar resultados.

---

## 0. ORDEM EXECUTIVA — LEIA ANTES DE MODIFICAR QUALQUER ARQUIVO

1. **INICIAR EM D0-A.** Não iniciar campanha nova nem implementar JointSelector. Localizar dados brutos, figuras e scripts da auditoria anterior; reproduzir os resultados por slice e a regra R19, sem confiar apenas no arquivo de parecer. Se D0-A não passar, D0-B/D1/D2/D3 ficam bloqueados.
2. **Não confundir dois fenômenos:** (i) recuperação ao ampliar o *espaço de ações*, removendo pisos; e (ii) recuperação por *melhor escolha dentro do mesmo espaço de ações*. Só o segundo sustenta diretamente **Joint Action Selection** como contribuição isolada.
3. **Não alterar simultaneamente modelos SQR, SLAs, espaço admissível, política de exploração, escalonador, τ, kernel, memória, γ, margens, ou mecanismo de atualização.** Comparação causal deve variar apenas a regra de seleção da ação final.
4. **Não tratar predição como verdade.** O seletor usa modelos online para escolher; o replay contrafactual é o avaliador independente. O seletor **nunca** recebe resultados dos replays como entrada.
5. **Não classificar ausência de evidência como SLA satisfeito.** Aplicar R19 e preservar `MISSING/PARTIAL/COMPLETE` por slice e janela.
6. **Não reabrir o hard gate FEASIBLE/MARGINAL/INFEASIBLE.** G4-B terminou em `STOP FINAL` (FFR=11,08%, FIR=9,52%). O classificador 3-way pode servir apenas a análises, nunca como condição de execução do seletor.
7. **Não presumir que `C=53` seja RBG, PRB, ou unidade do vetor de alocação.** Ler `resource_unit_manifest`, contrato IPC e implementação; distinguir cotas solicitadas, executadas e agendadas. O número e a unidade devem vir da configuração rastreada.
8. **Executar D0-A → D0-B → D1 → D2 → D3, com gates entre etapas.** Emitir relatório ao final de cada etapa, com evidências, hashes, dados e veredito `PASS`, `STOP` ou `INCONCLUSIVE`. D4/D5 exigem autorização científica explícita após D3.
9. **Não fazer tuning orientado pelo conjunto de teste.** Decisões de scoring, busca, seleção de estados, critérios e limiares são registradas *antes* dos replays de avaliação.
10. **Respeitar `AGENTS.md` e os documentos operacionais atuais do repositório.** Caso haja divergência, documentar e solicitar decisão; não substituir silenciosamente parâmetros congelados.

## 1. BASE CIENTÍFICA E O QUE É / NÃO É CONHECIDO

### 1.1. Artigo-base

*Tuerxun, A.; Nakao, A. “Safe RAN Slicing in O-RAN: Minimizing SLA Violations via Model-Based Reinforcement Learning”, IEEE CCNC 2026, DOI: 10.1109/CCNC65079.2026.11366589.*

No Algoritmo 1, o SQR mantém modelos de regressão quantílica por slice, seleciona propostas individuais com margens dependentes de incerteza e, quando `Σ a_i > C`, aplica uma projeção de capacidade. Há informação global no estado dos agentes: **não** alegar que a introdução de contexto global ou de um vetor conjunto seja inédita. A novidade em investigação está em **avaliar combinações diretamente na etapa de seleção**, em vez de selecionar individualmente e corrigir depois.

### 1.2. Evidências reportadas até 06/11/2026 — sujeitas ao novo gate D0-A

| Item | Classificação | Evidência / interpretação permitida |
|---|---|---|
| Rádio operacional, RLF≈0 sob carga | Resultado reportado da campanha | Violações podem surgir de disputa por recursos sem rádio ruim. |
| Violações crescem com carga; joelho aproximado λ≈4–5 na caracterização homogênea | Resultado reportado | Motiva investigar transição e congestionamento; **não** transfere automaticamente percentuais ao cenário heterogêneo. |
| Equal e Prop têm trade-offs | Resultado reportado | Nenhuma das duas opções estáticas domina. |
| Cinco replays: original `(18,17,18)` 0/5; ação com pisos `(37,13,3)` 0/5; ação fora dos pisos `(48,5,0)` 5/5 | **REPORTADO nos cinco estados/ações examinados; integridade causal e R19 ainda a revalidar em D0-A** | Demonstra a relevância da **admissibilidade** da ação; **não** prova que o SQR-HET perde as mesmas oportunidades por seleção independente/projeção. |
| URLLC: aumentar `a_u` entre 13 e 35 pouco mudou o miss nos replays citados; `a_u=0` colapsou | Resultado local | Não pressupor monotonicidade latência × recursos nem concluir que um piso universal foi identificado. |
| Fronteira F/M/I (P_F, P_M, P_I) | **CANDIDATA** | Não tratar valores como definitivos; “INFEASIBLE” significa apenas no espaço/critério testado. |
| G4-B FFR=11,08%, FIR=9,52%, gate exigia ambos ≤10% | **STOP FINAL** | Não declarar PASS retroativamente. |
| SQR-JC soft-risk / seleção conjunta | **NÃO TESTADO** | Sem evidência de superioridade, sem PILOT/MAIN validados. |

### 1.3. Perguntas de pesquisa que o protocolo deve distinguir

- **Q1 — Oportunidade:** existem estados válidos nos quais o SQR-HET viola, porém outra ação **no mesmo espaço admissível** recupera simultaneamente os SLAs?
- **Q2 — Mecanismo:** essas oportunidades se relacionam com o procedimento de seleção independente/projeção? Em particular, a projeção muda margens previstas ou a decisão executada?
- **Q3 — Exploração:** um seletor conjunto usando **os mesmos modelos treinados**, sem consultar o replay/oracle, identifica ações recuperadoras mais frequentemente que a seleção do SQR-HET?
- **Q4 — Operação:** em malha fechada, o ganho persiste sem piora sistemática de URLLC/mMTC, starvation, capacidade ou latência computacional?

**Critério científico central:** Q1+Q2 motivam a lacuna; Q3 demonstra que há um mecanismo viável; Q4 é necessário para reivindicar ganho de um controlador em operação simulada. Um ou cinco replays favoráveis isolados **não** bastam.

---

## 2. ESCOPO CONGELADO E HIPÓTESE ÚNICA

### 2.1. Candidato a contribuição

> **Seleção conjunta, consciente do risco previsto pelo SQR, de ações de alocação inter-slice diretamente sujeitas à restrição de capacidade, mantendo inalterado o backbone de aprendizado do SQR-HET.**

**H1 (a testar):** sob estados comparáveis e o **mesmo conjunto admissível de ações**, selecionar diretamente o vetor de alocação por meio das previsões existentes dos modelos SQR aumenta o atendimento simultâneo dos SLAs e o aproveitamento de oportunidades recuperáveis, relativamente à seleção individual seguida de projeção.

**H0 operacional:** a seleção conjunta não produz ganho relevante ou consistente; quaisquer recuperações decorrem sobretudo de mudanças no espaço de ações, inadequação do modelo, aleatoriedade, ou do método de avaliação.

**Não são contribuições nesta campanha:** pisos condicionados ao estado, novo regressor, novo kernel, nova função de loss, classificador hard F/M/I, novo simulador/cenário físico, predição neural, aprendizado multiagente, controlador de escalonamento por pacote, novo desenho TDD.

### 2.2. Braços comparados

- `B0 = SQR-HET`: política por slice + mecanismos originais de segurança, margem e fallback + projeção de capacidade. **Referência principal.**
- `B1 = SQR-HET + JOINT_SELECT`: mesmos modelos, atributos, hiperparâmetros, estados, buffers, atualizações, critério de evidência e restrições de recursos; **muda somente a escolha final do vetor de ação**. B1 não executa uma projeção corretiva posterior, porque só seleciona ações admissíveis.
- `B2 = ORACLE_TESTADO`: busca contrafactual via replay entre ações pré-registradas (potencialmente limitada). **Ferramenta de diagnóstico, nunca método de produção nem baseline online.**
- `B3 = ACTION_SPACE_RELAXED` (somente diagnóstico separado): amplia/remova pisos. **Não misturar** resultados de B3 com a comparação causal primária B0 × B1.

Uma ablação adicional, se necessária após D2, poderá testar outras **projeções sobre as mesmas propostas individuais**. Ela deve ser identificada como ablação de projeção e não acoplada silenciosamente ao B1.

### 2.3. Configuração experimental de referência — verificar, não reinventar

| Elemento | Valor reportado em `contexto_novo_chat_sqr_jc_contribuicao(1).md` | Regra |
|---|---|---|
| Plataforma | ns-3 / 5G-LENA + Python | Registrar versões e commits reais. |
| Ambiente físico | uma célula, UMa, raio 300 m, 30 dBm, 10 MHz | Congelado para comparação; 300 m é ponto calibrado, não raio normativo. |
| Scheduler / RLC | PF / RLC UM | Não alterar. |
| TDD | padrão congelado existente | Registrar padrão concreto do repositório. |
| Decisão | 100 ms | Distinguir decisão, execução no MAC e horizonte do KPI. |
| Slices | eMBB + URLLC + mMTC | Não usar as 3 eMBB do artigo como se fossem o mesmo cenário. |
| eMBB | throughput ≥3,5 Mbps | Parâmetro experimental. |
| URLLC | deadline 10 ms; miss ≤5%; *rolling* 1 s na G4-B | Não confundir deadline de pacote com decisão de 100 ms. |
| mMTC | PDR mínimo conforme `sla_manifest` | **Não inventar threshold.** |
| C, granularidade, mínimos | `resource_unit_manifest`/manifest de ações | **Não inventar PRBs/RBGs ou pisos.** |
| Chegadas, sessão, seeds e perfis | configs congeladas e artefatos da campanha | Recuperar fonte exata antes de usar. |

**Se algum dado estiver ausente ou contraditório:** registrar `BLOCKED_MISSING_MANIFEST`; não substituí-lo com um “valor razoável”. A configuração do artigo original, com três slices eMBB, não serve para preencher automaticamente lacunas da campanha heterogênea.

---

## 3. ONDE TRABALHAR NO REPOSITÓRIO

**Repositório de referência:** https://github.com/Eliothluy/Jussi_SBRC  
**Atenção:** a documentação pública do repo pode estar desatualizada frente aos experimentos de novembro; a fonte operacional definitiva é o **checkout local efetivo**, mais os manifests/dados nele armazenados. Não alegar ter auditado arquivos não inspecionados.

Mapeamento divulgado em `AGENTS.md`/README, **a confirmar no checkout**:

```text
Jussi_SBRC/
  AGENTS.md
  GUIA_IMPLEMENTACAO_SQR_O_RAN_PYTHON_NS3.md
  MODA/
    ns-3-dev/
      scratch/sqr/
        sqr-sim.cc                  # núcleo da simulação
        ...                       # scheduler, gates e wrappers existentes
    ns-o-ran-gym/
      src/nsoran/sqr/             # modelos/política SQR
      examples/sqr_run.py         # orquestração SQR citada no README
      tests/                      # testes Python
    docs/sqr/                     # contratos, manifests, auditorias e premissas
```

**Diretórios propostos para esta campanha (criar somente após inventário do checkout):**

```text
MODA/docs/sqr/jc_decision/
  PROTOCOL_FROZEN.md
  INVENTORY.md
  D0A_integrity_audit.md
  D0B_baseline_audit.md
  D1_projection_diagnostic.md
  D2_counterfactual_replay.md
  D3_offline_joint_selection.md
  GO_NO_GO.md
MODA/results/sqr_jc_decision/                   # resultados imutáveis por run_id
MODA/ns-o-ran-gym/src/nsoran/sqr/jc_selection.py # código B1 isolado
MODA/ns-o-ran-gym/examples/sqr_jc_*.py           # runners/diagnósticos
MODA/ns-o-ran-gym/tests/test_sqr_jc_*.py         # testes
```

Não presumir que esses scripts/diretórios já existem. São **interfaces de entrega propostas**. Respeitar a preferência existente por código novo isolado, sem alterar `scratch/rslaq/`, `scratch/our_paper/`, core global do gym ou `contrib/nr` sem autorização. Preservar os patches defensivos locais. Não modificar `AGENTS.md` ou fazer commit/push sem aprovação prevista no fluxo do projeto.

### 3.1. Inventário obrigatório do agente antes de qualquer alteração

Registrar em `INVENTORY.md`:

- `git rev-parse HEAD`, branch, `git status --short`, hash de configurações/manifests e versão de Python, ns-3, 5G-LENA.
- Caminho real do SQR-HET heterogêneo, classe que escolhe ações, implementação da projeção, função de atualização, atribuição de RNG e scripts de campanha/replay.
- Caminhos reais de `sla_manifest`, `resource_unit_manifest`, contratos IPC/ação/KPM, parâmetros do scheduler e estado do classificador R19.
- Identificação dos logs e checkpoints de simulações existentes, schemas, timestamps, seed IDs, horizonte por KPI e seus hashes.
- Disponibilidade de checkpoints determinísticos e trilhas de eventos exógenos reutilizáveis; teste de equivalência do prefixo.
- Disponibilidade de modelos quantílicos **que tratem corretamente** KPIs de sentido `maior é melhor` e `menor é melhor`; principalmente a cauda da URLLC.
- Inventário de testes existentes e comando operacional correto para o wrapper do ns-3 — `AGENTS.md` menciona `scratch/sqr/ns3_gate.sh`, mas confirmar no ambiente.

**Gate D0-B (após D0-A):** se SQR-HET ainda não existir, se não houver ação realmente aplicada no MAC, se a unidade de recurso for ambígua, ou se as métricas não tiverem validade temporal demonstrável, **STOP D0-B** e entregar lista objetiva de bloqueios. Não iniciar D1 presumindo essas condições.

---

## 4. CONTRATO DE DADOS E INVARIANTES DE VALIDAÇÃO

### 4.1. Identificador da unidade de análise

Criar uma chave composta estável:

```text
(run_id, simulator_commit, scenario_hash, seed, lambda,
 episode_id, decision_index, sim_time_ms)
```

A **unidade de inferência estatística** será o run/episódio/seed independente, **não** cada janela de 100 ms. Janelas vizinhas são autocorrelacionadas, especialmente com KPI URLLC agregado em 1 s.

### 4.2. Log por decisão (JSONL ou Parquet, sem arredondamento destrutivo)

Contrato mínimo ilustrativo — **schema novo proposto**, não nome de campo já verificado no código:

```json
{
  "run_id": "EXAMPLE_ONLY",
  "seed": 1,
  "lambda": 5,
  "episode_id": "EXAMPLE_ONLY",
  "decision_index": 120,
  "sim_time_ms": 12000,
  "policy": "SQR_HET",
  "state_hash": "...",
  "action_unit": "FROM_RESOURCE_UNIT_MANIFEST",
  "capacity_C": null,
  "a_raw_per_slice": [null, null, null],
  "a_after_uncertainty_margin": [null, null, null],
  "a_after_projection": [null, null, null],
  "a_sent_to_mac": [null, null, null],
  "a_effectively_applied": [null, null, null],
  "projection_triggered": null,
  "projection_delta_l1": null,
  "predicted_safe_before": [null, null, null],
  "predicted_safe_after": [null, null, null],
  "predicted_margin_after": [null, null, null],
  "predicted_uncertainty_after": [null, null, null],
  "slice_demand": [null, null, null],
  "slice_backlog": [null, null, null],
  "active_ues": [null, null, null],
  "sla_evidence": ["UNKNOWN", "UNKNOWN", "UNKNOWN"],
  "sla_outcome": [null, null, null],
  "kpi_window_start_ms": null,
  "kpi_window_end_ms": null,
  "model_checkpoint_hash": "...",
  "scenario_hash": "..."
}
```

`null` no exemplo significa **não preenchido**; nunca persistir `null` como dado medido sem um motivo `missing_reason`. Arrays devem possuir ordem explícita `[eMBB, URLLC, mMTC]`. Separar predição **feita em t** de KPI observado **após t**.

### 4.3. Regras R19 / SLA / tempo

- `COMPLETE`: todas as informações necessárias para o SLA daquela slice estão presentes, no horizonte correto.
- `PARTIAL`: algum requisito está ausente, atrasado, insuficiente ou censurado.
- `NO_ACTIVE_DEMAND`: slice sem demanda ativa. **Não converter automaticamente em `PASS`**; registrar separadamente e aplicar regra científica pré-definida.
- `INVALID`: erro de instrumentação, unidade, timestamp ou integridade.
- `ALL_SLA_OK` só é definido quando **todos os SLAs exigidos forem avaliáveis** e aprovados pelo `sla_manifest`. Em casos parciais reportar `UNKNOWN`. Separar análises estritas `ALL3_COMPLETE` e opcionais `ACTIVE_ONLY`, mas **nunca** usar active-only para afirmar all-three-OK; nenhuma convenção nova pode ser adotada pós-hoc para salvar 5/5.
- Para **URLLC 10 ms**, identificar deadlines por **pacote** e coortes de chegada. A janela móvel de **1 s** é agregação de observação, não duração do prazo. Uma ação mantida por 100 ms não pode ser creditada com a recuperação de pacotes que expiraram antes de `t`.
- Para análises causais, definir: horizonte da **ação** (100 ms), janela de **medição** (conforme manifesto) e efeito em pacotes que chegaram **após** a ramificação; reportar separadamente a métrica *rolling* que inclui histórico anterior.

### 4.4. Invariantes automáticos

```text
INV-01: todas as alocações executadas são inteiras e na unidade congelada.
INV-02: 0 <= a_i <= C e soma_i(a_i) <= C após aplicação.
INV-03: a ação executada corresponde ao controle recebido pelo MAC; divergências logadas.
INV-04: não somar PRBs, RBGs, slots ou percentuais como se fossem a mesma unidade.
INV-05: recursos requisitados/executados/agendados/ociosos são métricas separadas.
INV-06: modelo usado em t contém apenas observações disponíveis até t.
INV-07: replay com a MESMA ação reproduz estado e indicadores dentro da tolerância
        declarada; prefixo pré-ramificação deve ser bit-idêntico quando suportado.
INV-08: replay com ação alternativa difere apenas no tratamento especificado.
INV-09: ausência de demanda/evidência não conta como sucesso automático.
INV-10: todos os gráficos têm denominação de janela, seed, ação e versão.
INV-11: B0 e B1 usam MESMO espaço admissível no estudo primário.
INV-12: nenhuma saída do oracle/replay entra nas features do seletor B1.
```

---

## 5. D0-A — GATE DE INTEGRIDADE DAS EVIDÊNCIAS EXISTENTES (EXECUTAR PRIMEIRO)

> **ESTÁGIO ATUAL E OBRIGATÓRIO.** Este bloco foi acrescentado após o parecer `pesquisaCientifica.txt`. Seu objetivo não é gerar evidência a favor de Joint Selection. É determinar se os dados anteriores podem ser legitimamente reutilizados. **D0-A é o único estágio autorizado antes de seus testes passarem.** A inspeção e reexecução de testes de paridade/validação são permitidas; novas campanhas de otimização não são.

### 5.1. Política de evidências: diferenciar fonte de interpretação

O parecer **afirma** que verificou oito achados e detectou três inconsistências: (i) aparente discrepância entre throughput eMBB e all-SLA no gráfico de evitabilidade; (ii) percentuais R19 contraditórios; (iii) cortes de margens em figuras. **Não aceitar como verificados** os enunciados “100% correto”, “autenticidade comprovada”, “limite de Bayes irredutível”, “C=53 RBGs” ou “inviabilidade física matemática” sem trilha própria de código/dados. Um artigo ou norma citado no parecer sustenta conceitos gerais; **não substitui evidência sobre a execução particular**.

Criar `D0A_CLAIM_LEDGER.csv` com cada afirmação, seu status e caminho de comprovação:

| ID | Alegação reportada | Checagem exigida | Status inicial |
|---|---|---|---|
| A01 | 5/5 all-SLA-OK em `(48,5,0)` | Recalcular *por slice* a partir de traces e critérios R19 | `UNVERIFIED` |
| A02 | 0/5 em `(37,13,3)` mesmo com eMBB >3,5 Mbps em 3 casos | Identificar a(s) slice(s) realmente violada(s) e seus limiares | `UNVERIFIED` |
| A03 | 0/5 original `(18,17,18)` | Verificar trace original e replay A/A | `UNVERIFIED` |
| A04 | Regra R19: 24.482 total, 12.075 completos, 12.407 restantes | Recomputar por filtro, versão, seed e carga | `REPORTED` |
| A05 | Fig. 6: 86% λ2; 48% λ5; 25% λ6 | Reconstruir denominadores por λ nos arquivos-fonte | `REPORTED` |
| A06 | `C=53` = RBGs/RBs/PRBs? | Traçar unidade no manifest, IPC, scheduler, logs | `AMBIGUOUS` |
| A07 | URLLC a_u=13..35 com miss ~5–5,5%; a_u=0 com colapso | Recalcular coortes e horizontes; não atribuir causa TDD automaticamente | `REPORTED` |
| A08 | F/M/I cresce com λ | Verificar seleção e gate v2 STOP; manter valores candidatos | `CANDIDATE` |
| A09 | G4-B FFR=11,08%, FIR=9,52% | Localizar relatórios e definição dos denominadores | `REPORTED_STOP` |
| A10 | rádio saudável / RLF≈0 | Verificar rádio e KPI de todas as campanhas referenciadas | `REPORTED` |
| A11 | equal vs prop: trade-off | Reproduzir dados e distinguir cenário de D0-A | `REPORTED` |
| A12 | clipping das Fig. 4, 7 e rodapés | Inspecionar figuras originais; correção visual em branch separada | `REPORTED` |

**Convenção de status:** `VERIFIED_RAW`, `VERIFIED_REPLAY`, `REPORTED_ONLY`, `CONTRADICTED`, `UNVERIFIABLE`, `NOT_APPLICABLE`. Nunca usar “comprovado” com estado `REPORTED_ONLY`.

**Hierarquia para conflitos:** (1) configuração/hash efetivamente executados + código + dados brutos; (2) logs/traces e checkpoints identificados; (3) agregações recalculadas; (4) scripts de plotagem; (5) figuras já geradas; (6) descrições do relatório/parecer. Divergências devem permanecer no relatório; não escolher a versão mais favorável à hipótese.

### 5.2. Inventariar fontes e preservar cadeia de custódia

Localizar, **sem presumir que os nomes existam exatamente**, os seguintes materiais:

1. PNG/SVG/PDF das figuras `fig2_evitabilidade_espaco_acoes`, `fig6_vacuous_r19`, `fig3` URLLC, `fig4` aprendizado, `fig7` rádio; scripts que as produziram; datasets de entrada e relatórios de figura.
2. Manifest da campanha que gerou cada figura (`scenario_hash`, `run_id`, `seed`, carga, configuração TDD/MAC, granularidade da ação, versão R19, código do plotador).
3. Logs brutos de cada uma das **5 janelas** para cada uma das **3 ações**; vincular chave `(lambda, seed, decision_index, sim_time, action, replay_branch)`.
4. Traces de pacotes, filas, chegada/expiração, UE/slice e eventos de escalonamento necessários a recomputar eMBB, URLLC e mMTC.
5. `sla_manifest` com thresholds, população de UEs, função de agregação, tolerâncias e janelas; `resource_unit_manifest`; versão do oracle e scripts de replay.
6. Evidência de que os cinco prefixos e a política de continuação usados nas intervenções podem ser reconstruídos. Se não houver traces, anotar expressamente `UNVERIFIABLE` — não fazer engenharia reversa de valores de PNG.

Registrar caminhos absolutos/relativos, SHA-256, tamanho, hora de produção e commit **original** dos artefatos. Cópias de trabalho são imutáveis. **Nunca sobrescrever o dataset anterior nem os gráficos originais.** Se uma figura for corrigida, gerar nova versão com descrição da diferença.

Entregáveis imediatos:

```text
D0A_input_inventory.csv      # caminho, origem, sha256, run_id, scenario_hash, status
D0A_CLAIM_LEDGER.csv         # alegação → evidência → veredito
D0A_missing_evidence.md     # dados ausentes e impacto
```

Se o repositório não contiver o material, o agente pode preparar scripts e contratos de auditoria, mas deverá declarar `D0A_INCONCLUSIVE`, sem inventar os dados.

### 5.3. Auditar Fig. 2: cinco estados × três ações × três SLAs

**Matriz alvo:** **15 linhas** (5 estados × 3 ações), cada qual com avaliação de eMBB, URLLC, mMTC, máscara R19 e decisão conjunta. NÃO transcrever as 15 linhas com base na figura; gerar dos traces.

Estados citados no parecer: `λ4s3 t1812`, `λ5s1 t1955`, `λ6s3 t298`, além dos outros dois estados a localizar no dataset. Não presumir que `t1812` seja milissegundo: verificar se é índice de decisão, frame ou timestamp. Nenhum desses valores é um novo parâmetro do experimento.

Ações reportadas:

```text
A0_ORIGINAL             = (18,17,18)
A1_CONSERVATIVE_FLOORS  = (37,13,3)
A2_RELAXED_FLOORS       = (48,5,0)
```

**Execução para cada linha:**

1. Confirmar qual unidade representa cada número; sem unidade validada, interromper a auditoria física (`BLOCKED_UNIT`).
2. Reconstruir condições **pré-intervenção**: UEs, chegada, demanda e backlog por slice; hash dos estados e modelo/política de continuação.
3. Validar replay A/A: aplicar a ação original no mesmo snapshot e comparar outputs dentro de tolerância **pré-registrada**; verificar hash do prefixo e dos exógenos.
4. Recalcular **eMBB** exatamente conforme o manifest: população de UEs, estatística usada (média/percentil/mínimo), período, throughput Mbps e desigualdade `>=3.5`. Se painel superior mostra *média* e joint usa *mínimo*, marcar explicitamente `DIFFERENT_AGGREGATION`.
5. Recalcular **URLLC**: pacotes elegíveis, timestamps de entrada/saída, deadline **10 ms**, misses e denominador; computar tanto coorte causal pós-t como janela operacional rolling **1 s**, sem confundi-las. Verificar se a diferença entre `(37,13,3)` e `(48,5,0)` decorre realmente desse SLA.
6. Recalcular **mMTC**: pacotes gerados, recebidos, descartados, pendentes/censurados, PDR e threshold do `sla_manifest`, definindo o que conta no denominador.
7. Determinar `evidence_status` por slice: `COMPLETE`, `PARTIAL`, `NO_ACTIVE_DEMAND`, `INVALID`. Não interpretar `0/0`, NaN, lista vazia, `None` ou ausência de UE como 100% de sucesso.
8. Só então recomputar `joint_outcome` (`PASS`, `FAIL`, `UNKNOWN`) sob a definição congelada de **três SLAs avaliáveis**; registrar uma análise *active-only* separada se já houver regra científica para isso, **sem rotulá-la como all-three-OK**.
9. Comparar a classificação obtida com as 5×3 células da figura; gerar `mismatch_reason` e provas até o nível de pacote/UE, onde necessário.
10. Verificar especificamente por que eMBB >3,5 Mbps não implica sucesso conjunto em `(37,13,3)` nos três estados citados. Explicar qual slice falhou e **qual KPI/denominador** acionou a falha. Se o painel mede outro KPI, corrigir legenda do gráfico futuramente.

**Atenção crítica à ação `(48,5,0)`:** com `a_m=0`, distinguir (i) mMTC sem pacote/UE ativo; (ii) pacotes pendentes que não receberam recursos; (iii) entrega por fila anterior/efeito de janelas históricas; (iv) política de empréstimo de recursos; (v) atraso ou denominador censurado. Uma conclusão `PASS` para a mMTC só é legítima se existir **evidência positiva apropriada ao KPI definido**; ausência de amostras vira `NO_ACTIVE_DEMAND/UNKNOWN` conforme regra congelada, não `PASS` automático.

**Tabela obrigatória (esquema ilustrativo):**

```text
state_id,run_id,seed,lambda,decision_index,sim_time_ms,action_id,action_vector,
action_unit,model_hash,replay_prefix_hash,baseline_aa_pass,
eMBB_value,eMBB_threshold,eMBB_aggregation,eMBB_status,
URLLC_misses,URLLC_packets,URLLC_miss_rate,URLLC_rolling_miss_rate,URLLC_status,
mMTC_generated,mMTC_delivered,mMTC_pending,mMTC_pdr,mMTC_status,
active_ues_e,active_ues_u,active_ues_m,backlog_e,backlog_u,backlog_m,
evidence_e,evidence_u,evidence_m,joint_outcome,joint_original_label,
label_matches_figure,mismatch_reason,raw_trace_paths,trace_sha256
```

**Validação por invariantes:** `all-three-PASS ⇒ 3× evidence COMPLETE + 3× KPI PASS`; qualquer `UNKNOWN` não vira `PASS`; cada uma das três slices tem resultado explícito; `5 states × 3 actions = 15` se dados completos; `5/5` será recontado **somente depois** de todos os checks.

**Gate especial `REPLAY_5x3`:**

- `PASS`: resultados por slice e status de evidência reconstituídos, replay A/A fiel, labels da figura compatíveis ou erros corrigidos/revisados com trilha de mudança; 5/5 pode continuar ou ser revisto honestamente.
- `STOP`: falta de evidência positiva, unidade inconsistente, erro de classificação joint ou replay A/A inválido afeta a alegação de recuperabilidade.
- `INCONCLUSIVE`: dados brutos indisponíveis/insuficientes; não utilizar 5/5 como fato definitivo na motivação do controlador.

### 5.4. Auditar R19 e a população de janelas

O contexto consolidado reporta `24.482` janelas totais, `12.075` completas, `12.407` parciais/vacuous (soma exata). A fração não completa é `12.407/24.482 = 50,68%` aproximadamente. O parecer também reporta `λ2≈86%`, `λ5≈48%`, `λ6≈25%` **para a figura**. A porcentagem global **não** deve ser trocada por `88%`, nem `86%` deve virar `~100%` sem dados que sustentem tais números.

**Reprocessar do zero** a partir dos eventos por slice/UE/pacote — não a partir de barras de figura:

1. Identificar o universo: **qual rodada, versão, seeds, λ, janela e filtro de warm-up** geraram `N_total=24.482`.
2. Determinar para cada `(run_id, t)` o estado de evidência de cada SLA com a definição R19 original. Classificar conjuntamente em categorias mutuamente exclusivas, por exemplo: `COMPLETE`, `PARTIAL`, `NO_ACTIVE_DEMAND`, `INVALID` (documentar precedência em casos mistos).
3. Congelar a política de denominador: todas as janelas versus apenas aquelas com demanda simultânea, distinguindo-as; nunca misturar contagens de perfis/rodadas diferentes.
4. Produzir contagens e percentuais **por carga e seed**, com `n`, não apenas percentuais arredondados; conferir `sum(n_lambda,seed) == N_total` para a população comparável.
5. Calcular `complete + partial + no_active_demand + invalid == total` e `not_complete == total - complete`. Se a categorização detalhada não estiver disponível, não inventar a separação de `12.407`.
6. Investigar onde apareceram os percentuais `88%`, `~100%` e `51%`: versão antiga, filtro diferente, erro de cópia ou gráfico; registrar `denominator_id` e `protocol_version` de cada número.
7. Regerar `fig6` **somente** de tabela auditada, colocando nota de amostra e denominador. **Não alterar SLA ou filtro até que percentuais “batam”.**

**Saídas:** `r19_window_level.parquet`, `r19_by_seed_lambda.csv`, `r19_reconciliation.md`, `fig6_r19_audited.png`. Caso raw data não suporte reconstrução, preservar números reportados como `NOT_REPRODUCED`.

### 5.5. Auditar `C=53` e os recursos efetivamente controlados

**Pergunta física:** os elementos dos vetores `(18,17,18)` etc. representam **RB/PRB, RBG, quota lógica, orçamento por TTI ou outra abstração**? A soma 53, sozinha, não responde.

Procedimento obrigatório:

1. Inspecionar `resource_unit_manifest`, args do runner ns-3, contrato Python–C++/IPC, objeto de ação no agente e a função que aplica o controle na gNB/MAC.
2. Ler configuração real de numerologia, largura de banda, BWP/carrier, `rbPerRbg`, arredondamentos, TDD/slots DL e scheduler; anotar eventuais índices vs. quantidades.
3. Traçar o vetor de ação: **valor gerado em Python → serializado → recebido em C++ → alocação/limite no scheduler → RBs/RBGs efetivamente agendados**.
4. Capturar caso de teste por ação, com transformação de unidade, deltas de arredondamento, PRBs solicitados/executados, limites e slots DL com demanda. O mesmo símbolo `C` não pode designar simultaneamente RBs físicos e quota de alocação sem mapeamento explícito.
5. Comparar com o número de RBs/PRBs nominais *somente para explicar a implementação*, sem concluir autenticidade de execução a partir de comentários de outros testes do 5G-LENA.
6. **Não editar figuras para chamar `C=53` de “RBGs” antes dessa prova.** Se a unidade estiver incorreta, retificar rótulos e, caso o erro afete a execução, invalidar/reestruturar os experimentos afetados.

**Saídas:** `resource_units_trace.md`, `resource_units_mapping.csv`, `resource_units_assertions.log` e teste automatizado de conversão/limite por slot quando aplicável.

### 5.6. Auditar URLLC, hipóteses de TDD e janelas temporais

Sem mudar TDD, scheduler ou tráfego:

- Reproduzir o gráfico da insensibilidade URLLC comparando alocações e **o mesmo conjunto de coortes elegíveis**, registrando `arrivals`, `delivered_before_deadline`, `deadline_missed`, `pending/censored`.
- Medir timestamps `packet_arrival`, `deadline`, `DL_opportunity`, `scheduling`, `TX`, `RX` **se existirem nos traces**. Caso não existam, não inventar a causa física.
- Separar `deadline` de pacote **10 ms**, agregação `rolling 1 s` e decisão **100 ms**, além do horizonte do replay. Não atribuir à ação nova miss causado antes de `t` ou histórico herdado.
- Se houver oportunidade de DL, fila ou retransmissão, descrever associação; afirmar causalidade TDD versus scheduler apenas quando o desenho de eventos permitir isolar o fator.
- Na ausência de dados sub-slot, declarar `MECHANISM_NOT_IDENTIFIED` — o achado local de pouco ganho ao aumentar recursos permanece condicionado ao experimento, sem elevar a hipótese a limite universal.

Saídas: `urllc_replay_cohort.csv`, `urllc_timing_audit.md`, `fig3_urllc_audited.png` se dados suficientes.

### 5.7. Revisar as afirmações sobre F/M/I, aprendizado e rádio

Esta seção é uma **auditoria da interpretação**, não uma etapa de novo tuning:

- `INFEASIBLE`: relatar **“sem ação recuperadora confirmada no espaço discreto/suporte e sob o critério testado”**. Nunca inferir inviabilidade matemática universal só porque P_I cresce com λ. Valores F/M/I v2 permanecem `CANDIDATE` e G4-B permanece `STOP FINAL`.
- `MARGINAL`: classe operacional do método, não alegar equivalência formal a predição conforme ou à opção de rejeição de Chow sem algoritmo/calibração que justifiquem.
- FFR residual 11,08% não prova “limite irredutível de Bayes”, nem prova separação causal entre incerteza epistêmica e aleatória.
- RLF≈0 e SNR saudável suportam que o rádio não parece ser causa principal nas execuções auditadas; não equivalem a rádio “perfeito” nem a demonstração de uma única causa das violações.
- A curva de carga/latência não prova uso de fórmula de filas M/G/1 em um scheduler NR específico; evitar tal atalho sem pressupostos verificados.
- Resultados equal/prop devem estar vinculados aos seus próprios hashes, SLAs, cenários e métricas; não extrapolar automaticamente à política SQR-HET.
- As referências bibliográficas citadas em `pesquisaCientifica.txt` são **pistas de leitura, não validações verificadas aqui**; verificar metadados, conteúdo e pertinência antes de inseri-las no artigo.

Saída: `claims_interpretation_review.md` com colunas `claim`, `what_data_shows`, `what_data_does_not_show`, `allowed_manuscript_sentence`, `evidence_ref`.

### 5.8. Auditar gráficos sem confundir cosmética com integridade

Para cada figura original, registrar `input_sha256`, `plot_script_sha256`, `source_run_ids`, `schema_version`, `denominator_id`, título/caption original e problemas de clipping. Regerar em **diretório separado**, com eixo, unidade, legenda e caption compatíveis com o dado. Usar `bbox_inches='tight'` se adequado, mas verificar visualmente a saída (inclusive texto e margens) antes de aprovar. Correção de margem **não** altera dado; correção de label/denominador requer nova revisão dos resultados.

**Não bloquear D1 por problema puramente estético**, mas bloquear D1 se a figura não puder ser ligada aos traces ou se os rótulos evidenciam alteração de KPI/denominador.

### 5.9. Política para bugs encontrados

- `COSMETIC`: figura/legenda sem mudança de cálculo → gerar versão revisada, manter original.
- `TRACE_ONLY`: faltam logs, mas física/cálculo intactos → adicionar instrumentação sem alterar decisão; smoke test e revalidar checksums/paridade.
- `METRIC_BUG`: erro no denominador, status R19, janela temporal ou cálculo de KPI → corrigir em *nova versão*, recomputar **todos os braços afetados**; os labels e 5/5 anteriores ficam suspensos até revalidação.
- `SIM_BUG`: ação não aplicada corretamente, unidade errada, RNG não reproduzível → **STOP**, corrigir versão de simulador e reconstruir evidência causal; não mesclar runs antes/depois da correção.
- `MISSING_EVIDENCE`: sem traces/replay verificáveis → `INCONCLUSIVE`, relatar limitação e necessidade mínima de reconstrução. Só executar uma **campanha de regeneração autorizada e idêntica** quando não houver alternativa de reaproveitamento; não apresentar tal regeneração como reprodução bit-a-bit de runs ausentes.

### 5.10. Implementação mínima dos auditores (contratos, não scripts já existentes)

Implementar rotinas **isoladas da política** — não alterar o comportamento do controlador para auditar resultados históricos. Usar preferencialmente a infraestrutura Python e os parsers que o checkout já disponibiliza. Os nomes abaixo são **propostas**, a adaptar ao inventário real:

```text
sqr_jc_decision/audit/
  config/audit_sources.yaml           # fontes reais localizadas em D0-A
  audit_provenance.py                  # lista/hashes/manifest e conflitos
  audit_replay_matrix.py               # KPI/evidência por estado/ação/slice
  audit_r19.py                         # janelas e denominadores por λ/seed
  audit_resource_units.py              # Python → IPC → scheduler/MAC
  audit_urllc_timing.py                # coortes e métricas temporais
  audit_figures.py                     # dados agregados → figura auditada
  evaluate_gate_d0a.py                # critérios G0A-1..G0A-8
  tests/test_audit_replay_matrix.py
  tests/test_audit_r19.py
  tests/test_action_units.py
  tests/test_gate_d0a.py
```

**Contrato de entrada recomendado:** todas as rotinas recebem um manifesto de caminhos **verificados**, não assumem que os CSVs originais possuem nomes específicos e falham com diagnóstico explícito quando falta alguma coluna. O agente deve fazer um adaptador para os schemas históricos, preservando o arquivo original.

```yaml
# MODELO ILUSTRATIVO. Substituir somente após localizar as fontes no checkout.
audit_protocol_version: "D0A-v1"
source_checkout_commit: "TO_DISCOVER"
scenario_manifest: "TO_DISCOVER"
sla_manifest: "TO_DISCOVER"
resource_unit_manifest: "TO_DISCOVER"
original_figure_2: "TO_DISCOVER"
original_figure_6: "TO_DISCOVER"
replay_events: "TO_DISCOVER"
packet_traces: "TO_DISCOVER"
mac_allocation_traces: "TO_DISCOVER"
expected_state_count: 5
expected_action_count_per_state: 3
expected_global_r19_totals:   # somente teste de reconciliação; NÃO forçar coincidência
  total_reported: 24482
  complete_reported: 12075
  not_complete_reported: 12407
```

**Contrato do recomputador de SLA:** função pura que recebe eventos/traces de uma slice, timestamps, limiar, definição de janela e protocolo R19; retorna:

```python
# PSEUDOCÓDIGO: não é API verificada do repositório.
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class SliceVerdict:
    slice_name: str
    kpi_name: str
    value: Optional[float]
    denominator: Optional[int]
    evidence_status: str  # COMPLETE / PARTIAL / NO_ACTIVE_DEMAND / INVALID
    outcome: str          # PASS / FAIL / UNKNOWN
    evaluation_window: tuple
    source_trace_hash: str
    missing_reason: Optional[str] = None

def joint_verdict(verdicts: list[SliceVerdict]) -> str:
    # Resultado estrito: três SLAs com evidência COMPLETE são necessários.
    if len(verdicts) != 3:
        return "UNKNOWN"
    if any(v.evidence_status != "COMPLETE" for v in verdicts):
        return "UNKNOWN"
    return "PASS" if all(v.outcome == "PASS" for v in verdicts) else "FAIL"
```

**Atenção:** a função acima define a apresentação **estrita** de `ALL3_COMPLETE`, não substitui a semântica operacional original do dataset. Se o manifest original tiver outra política legítima para slices sem demanda, computar a respectiva métrica **separadamente**, com nome e denominador diferentes. Não mudar retroativamente a regra para preservar os cinco resgates.

**Testes sintéticos obrigatórios do próprio auditor (não são resultados da pesquisa):**

1. Três slices `PASS` + `COMPLETE` → `joint=PASS`.
2. Duas `PASS` + uma `FAIL`, todas `COMPLETE` → `joint=FAIL`.
3. eMBB acima do limiar, URLLC `FAIL` → `joint=FAIL`.
4. mMTC sem pacote (`0/0`) + outras duas `PASS` → `joint=UNKNOWN`, jamais `PASS` por vacuidade.
5. URLLC com packets incompletos/censurados → denominador e status explícitos.
6. R19 de contagens sintéticas não permite total diferente da soma de categorias.
7. Unidades `RB`, `PRB`, `RBG` e quota lógica diferentes não são convertidas implicitamente.
8. Alocação `C+1` ou negativa é rejeitada, não arredondada silenciosamente.
9. Replay A/A divergente → `REPLAY_INVALID`, não `RECOVERABLE`.
10. Falta de qualquer trace indispensável → `INCONCLUSIVE`, não string vazia ou sucesso.

**Comandos propostos após criação e validação das rotinas** — não supor que existam antes da implementação:

```bash
# São exemplos de CLI a IMPLEMENTAR, não comandos garantidos do repositório.
python -m sqr_jc_decision.audit.audit_provenance --config audit_sources.yaml
python -m sqr_jc_decision.audit.audit_replay_matrix --config audit_sources.yaml
python -m sqr_jc_decision.audit.audit_r19 --config audit_sources.yaml
python -m sqr_jc_decision.audit.audit_resource_units --config audit_sources.yaml
python -m sqr_jc_decision.audit.audit_urllc_timing --config audit_sources.yaml
python -m sqr_jc_decision.audit.evaluate_gate_d0a --config audit_sources.yaml
pytest -q sqr_jc_decision/audit/tests
```

Caso a estrutura do repositório impeça essa invocação, documentar no `INVENTORY.md` os comandos verdadeiros, o ambiente usado e os artefatos que cada comando gera. **Não executar um falso runner por aderência ao exemplo.**

**Contrato de `D0A_GATE.json` (proposta):**

```json
{
  "gate": "D0A",
  "version": "D0A-v1",
  "status": "INCONCLUSIVE",
  "criteria": {
    "G0A-1": "UNKNOWN",
    "G0A-2": "UNKNOWN",
    "G0A-3": "UNKNOWN",
    "G0A-4": "UNKNOWN",
    "G0A-5": "UNKNOWN",
    "G0A-6": "UNKNOWN",
    "G0A-7": "UNKNOWN",
    "G0A-8": "UNKNOWN"
  },
  "evidence_inventory_file": "D0A_input_inventory.csv",
  "replay_matrix_file": "D0A_replay_5x3.csv",
  "r19_summary_file": "r19_by_seed_lambda.csv",
  "resource_audit_file": "resource_units_trace.md",
  "blockers": ["PREENCHER COM BLOQUEIOS REAIS"],
  "next_stage_authorized": false
}
```

O JSON acima é **um exemplo de estado inicial bloqueado**, não um gate já executado. O campo `next_stage_authorized` somente poderá ser `true` se a avaliação automatizada e a revisão metodológica permitirem `D0A_PASS`. Para a entrega, o agente deverá indicar a localização real dos artefatos em vez de manter placeholders.

### 5.11. GATE D0-A (NÃO NEGOCIÁVEL)

Marcar separadamente:

```text
G0A-1  5×3 matrix tem proveniência de dados e métrica por slice.    PASS/FAIL/UNKNOWN
G0A-2  Os 5 snapshots originais têm A/A válido ou limitação declarada. PASS/FAIL/UNKNOWN
G0A-3  Labels all-3-SLA-OK e R19 são matematicamente consistentes.  PASS/FAIL/UNKNOWN
G0A-4  mMTC a_m=0 tem evidência/ausência de evidência explícita.     PASS/FAIL/UNKNOWN
G0A-5  R19 global e por λ têm denominadores/versões identificados. PASS/FAIL/UNKNOWN
G0A-6  `C` e os vetores de ação têm unidade rastreada até o MAC.   PASS/FAIL/UNKNOWN
G0A-7  KPI URLLC respeita coorte, timestamp, deadline e rolling.  PASS/FAIL/UNKNOWN
G0A-8  O parecer não é tratado como dado bruto; interpretações marcadas. PASS/FAIL/UNKNOWN
```

**`D0A_PASS`:** todos os itens que afetam a validade das métricas e a causalidade têm `PASS`. Divergências cosméticas registradas podem permanecer abertas. A conclusão 5/5 **não precisa sobreviver** para o gate passar: se a recalculação corrigir honestamente os resultados e os dados/replay permanecerem íntegros, o gate é um PASS de integridade **com resultado científico revisado**. Atualizar a motivação correspondente. O agente não deve “forçar” o antigo 5/5.

**`D0A_STOP`:** erro causal/unidade/R19 crítico sem correção confiável, replay não determinístico ou rótulo de violação sistematicamente inválido. D1/D2/D3 bloqueados até correção e nova auditoria.

**`D0A_INCONCLUSIVE`:** evidência essencial ausente; D1/D2/D3 bloqueados. Criar plano mínimo de obtenção/regen para a pesquisadora aprovar, sem improvisar simulações.

**Entrega obrigatória:** `D0A_integrity_audit.md`, `D0A_CLAIM_LEDGER.csv`, `D0A_replay_5x3.csv`, `r19_by_seed_lambda.csv`, `resource_units_trace.md`, `D0A_GATE.json`, relatórios de testes e referências/hashes dos dados. Se não houver resultados brutos, gerar a estrutura e declarar a impossibilidade, sem linhas inventadas.

---

## 6. D0-B — BASELINE, INSTRUMENTAÇÃO E CONGELAMENTO (SEM CAMPANHA NOVA)

### Tarefas

**Pré-condição:** `D0A_GATE.json` com `D0A_PASS`, artefatos e hashes verificáveis. Se o resultado anterior tiver sido corrigido, propagar versão revisada para todas as hipóteses e figuras; não usar os rótulos antigos.

1. Ler `AGENTS.md`, documentação de premissas e os três arquivos de base recebidos no chat; localizar no repositório a versão atual de `sla_manifest`, `resource_unit_manifest` e protocolo R19.
2. Reconstruir os parâmetros reais do cenário heterogêneo e o contrato Python↔C++; **nada de valores pressupostos**.
3. Auditar o SQR-HET contra o Algoritmo 1 do CCNC 2026: Matérn/quantis, memória, τ, incerteza, fallback, ação individual, margem, projeção, atualização virtual, timing.
4. Auditar semântica dos KPIs heterogêneos. Para eMBB e mMTC, maior KPI tipicamente é melhor; para URLLC miss rate, menor é melhor. **Não** transformar um *quantil inferior* da taxa de miss em um limite superior conservador por mera subtração. Verificar cauda/orientação adequada **já existente em B0**, sem mudar modelo nesta campanha.
5. Auditar a existência e fidelidade do replay (seeds, snapshots, agendas exógenas, RNG, checkpoint do modelo). Reexecutar uma ação idêntica em múltiplos checkpoints.
6. Identificar dados existentes suficientes para D1 e para a amostragem D2; evitar duplicar campanhas.
7. Congelar critérios e protocolo em `PROTOCOL_FROZEN.md` **antes** de investigar os estados escolhidos e antes da primeira execução de D2; o resultado revisado de D0-A serve como entrada, nunca como conjunto de teste de D3.

### Entrega e decisão

- `D0B_baseline_audit.md`, `INVENTORY.md`, `PROTOCOL_FROZEN.md`, lista de testes executados, provas de paridade e tabela de divergências.
- `PASS`: manifests e instrumentação confiáveis, SQR-HET bem definido, replay repetível.
- `STOP`: ausência estrutural impeditiva (ex.: não há controle real aplicado); corrigir apenas bug/instrumentação e repetir D0.
- `INCONCLUSIVE`: não é possível verificar a integridade; não avançar.

---

## 7. D1 — DIAGNÓSTICO DE SELEÇÃO INDEPENDENTE E PROJEÇÃO

**Objetivo:** caracterizar o comportamento do **SQR-HET real**, sem introduzir B1.

### 7.1. Instrumentação

Em cada decisão, registrar:

- estado *pré-decisão* e hash; IDs de run/seed/carga;
- `a_raw`, `a_after_margin`, `a_after_projection`, `a_sent_to_mac`, `a_applied`;
- soma das solicitações antes da projeção; flag e magnitude da correção;
- previsões do modelo para cada ação individual **antes** e **depois** da correção, com suporte/OOD, incerteza e máscara de SLA;
- situação de cada slice: demanda, backlog, atividade e SLAs válidos;
- resultado após execução, com horizonte causal e janela de observação explícitos.

**Nota crítica:** uma proposta `a_raw` com `Σa>C` **não é uma ação conjunta executável**; uma previsão “segura antes da projeção” não pode ser usada como contrafactual físico. A análise antes/depois é diagnóstica. A prova causal depende de comparar a ação realmente executada com **outra ação factível** em D2.

### 7.2. Estatísticas obrigatórias

1. `P_projection = N_decisões_com_projeção / N_decisões_válidas`.
2. Distribuição de `Σa_raw/C` e `||a_raw - a_projected||_1`.
3. Fração de ações com degradação de margens previstas pela correção, sem atribuição causal automática.
4. Taxa de violação por slice/joint, com máscara R19, **estratificada** por projeção, carga, demanda, backlog e seed.
5. Matriz: `projection_active` × `joint_SLA_outcome` (`PASS/FAIL/UNKNOWN`).
6. Comparação entre decisões com e sem projeção em estratos semelhantes de carga e demanda; reportar confundimento residual.
7. Evidência de que diferenças entre ação solicitada e MAC aplicado são logadas e explicadas.

### 7.3. Saídas

```text
D1_projection_diagnostic.md
projection_summary.csv
projection_per_seed.csv
projection_events.parquet
fig_projection_rate_by_lambda.png
fig_before_after_prediction.png
fig_outcomes_by_projection_and_load.png
```

### 7.4. Gate

- **D1 não demonstra causalidade por si só.** `PASS` significa que logs são íntegros e existem estados relevantes a investigar; ou, mesmo sem projeção, existe uma hipótese explícita de má escolha independente que D2 possa testar.
- `STOP/RETHINK`: ausência de evidência para uma lacuna atribuível à seleção **e** impossibilidade de formar um conjunto útil de estados.
- **Não selecionar apenas violações com projeção.** D2 precisa de controles sem projeção e/ou sem violação.

---

## 8. D2 — REPLAYS CAUSAIS PRÉ-REGISTRADOS: EXISTE RECUPERAÇÃO NO MESMO ESPAÇO?

### 8.1. Amostragem de estados — exploratória, não cherry-picking

**Proposta inicial:** até **60 estados distintos** de decisões do B0, vindos de **pelo menos 3 seeds independentes** e cargas `λ∈{2,4,5,6,7}` **se presentes no conjunto atual**. Dar atenção a λ 5 e 6, mas incluir controles.

Estratos desejados:

- `S1`: projeção + violação observada;
- `S2`: violação sem projeção;
- `S3`: projeção sem violação;
- `S4`: sem projeção e sem violação.

Não impor preenchimento artificial dos estratos: registrar quantidade disponível, fração da população, taxa de inclusão e razões de exclusão. Selecionar programaticamente com seed de amostragem **pré-fixada e registrada**; evitar estados duplicados e janelas altamente sobrepostas. Agrupar inferência por run/seed. As quantidades 60 e 3 são **orçamento exploratório**, não tamanho amostral estatisticamente suficiente.

### 8.2. Replay correto

Para cada estado `s_t`:

1. Restaurar **exatamente** o checkpoint/prefixo. Fixar seed, UEs, chegadas, tráfego, canal e estado pré-decisão do scheduler. Verificar hashes de prefixo.
2. Reexecutar **a mesma ação B0**: controle de paridade A/A. Se divergir, **bloquear o estado** e investigar determinismo.
3. Aplicar **uma** ação candidata `a'` admissível (mesmos pisos e limites de B0), apenas no período de decisão alvo. Após ele, utilizar **uma política de continuação pré-definida e idêntica para todos os ramos**, sem permitir que o seletor aprenda com futuros contrafactuais.
4. Preservar *common random numbers* para processos exógenos; se RNG endógeno se desalinhar após a intervenção, registrar e usar trilhas exógenas pré-geradas ou outro mecanismo verificável. Não declarar trajetórias bit-idênticas depois da ação, pois elas podem legitimamente divergir.
5. Medir SLAs na janela/horizonte pré-definidos. Para URLLC: distinguir coorte de pacotes com chegada a partir de `t` e respectivo prazo de 10 ms, de agregado *rolling* de 1 s com histórico prévio.
6. Salvar estado, ação, orçamento de busca, resultados por KPI, evidência R19, seeds e checksums do replay.

**Validade causal local:** a comparação A0 × A1 só é válida se o prefixo e os processos exógenos forem controlados, se a intervenção for a única diferença relevante e se os resultados tiverem evidência completa. Mesmo assim, a inferência fica limitada ao estado, ao horizonte e às ações testadas.

### 8.3. Espaços e busca

**Primário — `A_common(s)`**: conjunto de vetores inteiros admissíveis para **B0 e B1**, mesma capacidade, granularidade, limites e pisos *efetivamente vigentes*. Se o B0 original não possui pisos, **não criar pisos para o comparativo**. Se o baseline real usa pisos de uma campanha experimental, documentar origem e verificar se pertencem ao controlador ou apenas ao desenho do replay.

**Secundário — `A_relaxed(s)`**: espaço sem pisos experimentais; usar **somente para demonstrar dependência da admissibilidade**, nunca para provar a superioridade da coordenação conjunta.

A busca de contrafactuais deve ser determinística, pré-registrada e computacionalmente limitada. **Configuração inicial proposta:** orçamento máximo de **24 alternativas por estado** no `A_common`, além da ação B0 de paridade, combinando:

- alternativas estruturadas de redistribuição de recursos entre pares de slices;
- vizinhança da ação aplicada, respeitando a granularidade congelada;
- uma amostra aleatória reprodutível do espaço comum, com distribuição registrada;
- ações `equal/prop` apenas se **admissíveis**, com implementação/cotas idênticas às referências.

Se viável, fazer enumeração exaustiva em um **subconjunto previamente escolhido** e identificá-lo separadamente. Com busca parcial, usar sempre “**recuperação encontrada**” ou “**não recuperada nas alternativas testadas**”; **nunca** “impossível” ou “ótimo global”. Registrar `N_actions_tested`, cobertura e procedimento de busca, inclusive quando encontrar recuperação cedo.

### 8.4. Rotulagem sem falsa certeza

Para um estado de falha B0:

```text
RECOVERABLE_WITHIN_COMMON_SPACE:
    existe ao menos uma ação TESTADA em A_common que produz ALL_SLA_OK válido
NOT_RECOVERED_IN_TESTED_SET:
    nenhuma ação testada no espaço comum recuperou
RECOVERABLE_ONLY_AFTER_RELAXATION:
    nenhuma recuperação encontrada no comum e alguma encontrada no relaxado
UNKNOWN:
    replay, causalidade ou evidência de SLA incompletos
```

`NOT_RECOVERED_IN_TESTED_SET` **não** é sinônimo de inviabilidade física. `RECOVERABLE_ONLY_AFTER_RELAXATION` também depende do limite da busca no espaço comum.

### 8.5. Métricas D2

- `RecoverabilityFoundRate`: fração das violações B0 **válidas e testadas** que possuem pelo menos uma recuperação confirmada em `A_common`.
- `RelaxationOnlyFoundRate`: fração com recuperação encontrada **somente** no espaço relaxado, sob o orçamento declarado.
- Distribuição por carga, slice originalmente violada, projeção, seed, demanda e distância entre `a_B0` e `a_recovery`.
- Número de SLAs que eram válidos antes/depois; gravidade da violação; apoio/OOD das ações testadas.
- Cobertura e custo da busca por estado; casos `UNKNOWN`/excluídos apresentados explicitamente.

### 8.6. Veredito de D2

- **GO científico preliminar:** há recuperações **replicadas em estados de múltiplas seeds/episódios independentes**, dentro do **mesmo espaço admissível**. A relevância da frequência deve ser julgada com os denominadores, cobertura e incerteza, e não só pelo número absoluto.
- **RETHINK para `state-conditioned floors`:** recuperações aparecem principalmente fora dos pisos, sem evidência suficiente no espaço comum.
- **INCONCLUSIVE:** amostra, replay, R19 ou busca não permitem decidir; entregar limite da evidência, sem extrapolar.

**Não iniciar D3 caso D2 mostre apenas o ganho de retirar pisos e nenhuma justificativa independente para a seleção conjunta.**

---

## 9. D3 — PROTÓTIPO DE SELEÇÃO CONJUNTA OFFLINE, SEM TREINAR NOVO MODELO

### 9.1. Objetivo

Demonstrar que o **modelo disponível no instante da decisão**, sem acesso ao futuro ou à rotulagem do replay, consegue escolher vetores em `A_common` que aproveitem oportunidades de recuperação. **Isso ainda não demonstra ganho em malha fechada.**

### 9.2. Condições de isolamento

- Carregar **o mesmo checkpoint SQR-HET** e as mesmas features disponíveis antes de `t`.
- Mesmos modelos por slice, treinamento prévio, memória, τ, γ, incerteza, quantis e unidade de ação; não acrescentar q25/q75, detector 3-way ou novas perdas, salvo se já fizerem parte do B0 congelado e forem necessários para sua fidelidade.
- B0 e B1 têm **idêntico** `A_common(s)` e regra de avaliação de SLA.
- O seletor **não executa** o oracle nem consulta banco de resultados de replay.
- O ranking e os critérios de desempate devem ser **pré-registrados antes de observar o conjunto de teste**.
- Avaliar sobre **seeds/episódios de teste disjuntos dos usados para definir ranking, normalização e orçamento de busca**. Priorizar separação por seed/episódio, nunca por janela aleatória sobreposta.

### 9.3. Interface mínima proposta

```python
# PSEUDOCÓDIGO/CONTRATO — não é API verificada do repositório.
class JointSelector:
    def select(self, observation_t, sqr_models, action_manifest,
               sla_manifest, model_snapshot_t):
        """Return one feasible joint action; never read replay outcomes."""
        # 1. Establish active/assessable slice mask using only t-known data.
        # 2. Generate candidates in the SAME action space as SQR-HET.
        # 3. For each slice and candidate amount, reuse the existing
        #    SQR predictor + uncertainty / safety orientation.
        # 4. Rank joint vectors by a pre-registered aggregate.
        # 5. Apply deterministic tie-breaking and runtime budget.
        # 6. Return (action, trace, fallback_reason).
        raise NotImplementedError
```

### 9.4. Função de comparação de ações — uma proposta **a congelar em D0-B**

Usar as saídas conservadoras existentes do **SQR-HET**, expressas como **margens de SLA com sentido unificado** (`positiva=melhor`, `negativa=pior`). Seja `g_i(s,a_i)` a margem normalizada e conservadora **validada** do modelo da slice `i`; sua construção e calibração devem ser a mesma do B0, salvo o cálculo agregado sobre o vetor.

```text
Para cada a = (a_e, a_u, a_m) ∈ A_common(s):
    critical(a) = min_i g_i(s, a_i)   # sobre SLAs avaliáveis/ativos
    efficient(a) = soma_i utility_original_i(s, a_i)
Selecionar lexicograficamente:
    1º maior critical(a)
    2º maior efficient(a)
    3º menor soma_i a_i
    4º desempate determinístico pelo vetor a
```

Esse *ranking* é uma **proposta de implementação**, não um resultado ou fórmula do artigo-base. Se uma ação não for considerada suficientemente suportada pelos modelos, aplicar **a mesma política de incerteza/fallback do SQR-HET**, registrando decisão e frequência, sem filtrar em função de um hard gate F/M/I.

**Restrição crucial:** para KPIs `menor é melhor`, como URLLC miss rate, a margem conservadora deve considerar a **cauda superior do KPI desfavorável** ou uma regressão apropriada da margem. Não usar `limiar − quantil inferior de miss` como margem pessimista. Se B0 **não** tiver essa interpretação validada, interromper D3 e registrar o bloqueio; corrigir isso altera o backbone e exigiria um novo protocolo para comparação justa.

### 9.5. Busca computacional

- **Primeira opção:** enumerar exatamente `A_common(s)` quando a cardinalidade e custo de inferência permitirem. Para 3 slices, o número de triplas inteiras com `Σa≤C`, sem outras restrições, é `binom(C+3,3)`; **C é apenas símbolo até a auditoria do manifest**.
- Pré-computar `g_i(s,a_i)` e `utility_i(s,a_i)` **uma vez por quantidade candidata e por slice**, reutilizando valores na avaliação das triplas. O custo de inferência kernelizada não deve ser repetido ingenuamente para cada tripla.
- Caso não caiba no orçamento, utilizar busca determinística pré-registrada (p.ex. beam/search nos candidatos precomputados) e registrar o conjunto efetivamente explorado; incluir a ação B0 como candidata para evitar omissão por cobertura.
- Medir p50/p95/p99 e pior caso de tempo de decisão **sem inferir automaticamente latência real de xApp** a partir do benchmark isolado.

### 9.6. Teste offline corretamente pareado

O conjunto de avaliação de D3 deve incluir **tanto** estados em que B0 violou quanto estados em que B0 passou. Caso contrário, o experimento só mede *resgates* e ignora ações B1 que estragam situações anteriormente corretas.

Para cada estado reservado:

1. Congelar observação `s_t`, checkpoint do modelo e `A_common`.
2. Obter `a_B0` pela política fiel e `a_B1` pelo `JointSelector` sem consultar resultado.
3. Reexecutar **ambas** as ações em replay controlado, com mesmo estado e exógenos, coletando `PASS/FAIL/UNKNOWN`.
4. Usar o `ORACLE_TESTADO` apenas para marcar se existia recuperação confirmada entre as alternativas exploradas; nunca usá-lo na decisão de B1.
5. Preservar casos sem evidência separadamente e reportar quantas ações são OOD/in-support.

### 9.7. Métricas D3 — evitar comparações tautológicas

**Métrica principal do experimento pareado:**

`Δ_allOK = P(ALL_SLA_OK | B1) − P(ALL_SLA_OK | B0)` nos **mesmos estados válidos** (reportar estimativa estratificada e, quando possível, ponderada pela prevalência do estrato na população D1).

**Métricas explicativas:**

- `RescueRate = N(B0=FAIL, B1=PASS, recuperação confirmada no espaço comum) / N(B0=FAIL, recuperação confirmada no espaço comum)`.
- `HarmRate = N(B0=PASS, B1=FAIL) / N(B0=PASS)`.
- `NetRescues = N(B0=FAIL, B1=PASS) − N(B0=PASS, B1=FAIL)` e valor por 1.000 decisões elegíveis, **ponderado** quando houver amostragem estratificada.
- Gravidade de violações, estatísticas por slice, OOD, diferença de recursos alocados, capacidade e tempo de inferência.
- Taxa de eventos em que o resultado de B1 não é avaliável (`UNKNOWN`) — não converter em PASS.

**Evitar a métrica incorreta:** comparar `RescueRate_B1 > RescueRate_B0` **somente** entre estados definidos como `B0=FAIL`; nessa seleção, B0 tem resgate zero por construção, gerando comparação tautológica. Use `Δ_allOK`, `RescueRate_B1` condicional, `HarmRate` e denominadores transparentes.

### 9.8. Critério de D3 (antes do PILOT)

A decisão deve combinar:

- recuperação **dentro do espaço comum** observada em D2, com múltiplas seeds;
- melhoria pareada positiva no conjunto de teste, **sem** crescimento relevante de danos a estados anteriormente corretos;
- resultados não explicados apenas por nova liberdade de ações, OOD ou ausência de SLA;
- ausência de custo de inferência incompatível com o período de controle de 100 ms;
- efeito com relevância prática e incerteza documentadas por **unidade independente** (run/seed), sem usar janelas autocorrelacionadas como amostras independentes.

**Valores numéricos para relevância mínima e não-inferioridade** devem ser fixados por pesquisadores **antes** do teste em `PROTOCOL_FROZEN.md`. Se não houver justificativa científica para thresholds, declarar o gate **exploratório** e não chamar D3 de comprovação estatística final. É proibido ajustar limiares pós hoc para transformar `STOP` em `PASS`.

**Se passar:** autorizar apenas proposta de implementação do PILOT, ainda dependente da decisão do responsável científico. **Se falhar:** declarar `STOP-JOINT`, retornar ao resultado sobre espaço de ações/pisos como hipótese alternativa; não iniciar tuning aberto.

---

## 10. D4 E D5 — PLANEJADOS, PORÉM BLOQUEADOS

> **NÃO EXECUTAR AGORA.** Esses estágios aparecem para completar o pipeline, mas sua autorização depende das conclusões de D0-A–D3. O agente deve encerrar a entrega inicial com `GO_NO_GO.md`, não iniciar D4 automaticamente.

### D4 — PILOT em malha fechada

- Métodos: B0 (SQR-HET) e B1 (mesmo SQR + apenas seleção conjunta).
- Cargas: `λ={2,5,6}`; **3 seeds pareadas**; **300 s** por execução (proposta consolidada).
- Treinamento online em ambos: mesmo algoritmo, mesma inicialização quando aplicável, mesmas observações e regras de atualização; após ações divergirem, as trajetórias endógenas **podem** divergir naturalmente.
- Avaliar SLA conjunto e por slice, starvation, capacidade, fallback, OOD, estabilidade temporal, latência de decisão e interferência com scheduler/TDD.
- Condições de STOP: crash/inconsistência, violação de capacidade, degradação sistemática material de URLLC/mMTC, custo de decisão operacionalmente inviável ou ausência de benefício exploratório justificado.
- Não usar o PILOT para ajustar hiperparâmetros em ciclos abertos. Caso metodologia exija mudanças, registrar novo protocolo e separar execução exploratória da confirmação.

### D5 — MAIN confirmatório

- Condição: D4 aprovado e protocolo estatístico pré-registrado.
- Proposta de desenho: **10 seeds pareadas**, **600 s** por execução, perfil `2 → L_O → 2`.
- **`L_O` ainda deve ser validado/congelado no manifest antes do MAIN. Não inventar valor.**
- Principal: diferença de violações evitáveis confirmadas por replay em amostra pré-registrada das duas políticas, com definição clara do universo elegível; acompanhar `Δ_allOK` e trade-offs por slice.
- Análise: diferenças pareadas por seed/run; intervalos de confiança por clusters independentes e análise de sensibilidade, sem pseudorreplicação de janelas de 100 ms.
- Se 10 pares forem insuficientes, reportar inconclusão; não expandir até obter significância sem novo planejamento e justificativa de poder/amostra.

---

## 11. TESTES AUTOMÁTICOS DE ACEITE

Implementar/estender testes, **sem depender de visualização humana**, para pelo menos:

```text
TEST-01  Schema de logs versionado, seed e hashes presentes.
TEST-02  Soma de recursos <= C e unidade consistente em cada etapa.
TEST-03  Projeção registrada com propostas pré e pós margem e pós projeção.
TEST-04  Mesmo estado + mesma ação -> resultado reprodutível (dentro da tolerância congelada).
TEST-05  Prefixo anterior à intervenção não sofre alteração.
TEST-06  Contrafactual muda APENAS a ação do intervalo especificado.
TEST-07  Timings: sem vazamento de KPI futuro nas features do agente.
TEST-08  URLLC: deadline 10 ms ≠ rolling 1 s ≠ decisão 100 ms.
TEST-09  Janela PARCIAL / NO_ACTIVE_DEMAND não vira ALL_SLA_OK silenciosamente.
TEST-10  A_common idêntico para B0 e B1; ação B1 sempre admissível.
TEST-11  B1 não acessa resultados do replay/oracle.
TEST-12  Quantis e margens conservadoras respeitam o sentido do KPI.
TEST-13  Ordenação e desempates determinísticos; mesma entrada -> mesma ação.
TEST-14  Busca limitada respeita orçamento e relata cobertura.
TEST-15  B0 permanece compatível com o SQR-HET anterior, sem regressão.
TEST-16  D3 avalia rescues E harms; não reporta métrica tautológica.
TEST-17  Qualquer alteração de cenário/manifests invalida comparação anterior.
TEST-18  D4 e D5 ficam inativos enquanto GO_NO_GO não autorizar.
TEST-19  5 estados x 3 ações: outcomes por slice e status R19 concorrem com joint label.
TEST-20  mMTC com zero recursos não pode virar PASS apenas por ausência de pacote.
TEST-21  N_total == N_COMPLETE + N_PARTIAL + N_NO_ACTIVE + N_INVALID (categorias mutuamente exclusivas no denominador escolhido).
TEST-22  Reconstrução das figuras usa os MESMOS dados/denominadores do texto.
TEST-23  Se replay A/A divergir, estado não pode receber label RECOVERABLE.
TEST-24  Saída joint-OK exige status COMPLETE para todas as slices exigidas.
TEST-25  Unidade de ação foi rastreada até o MAC e respaldada pelo resource_unit_manifest.
TEST-26  Comparação principal não mistura B0 em espaço restrito com B1 em espaço relaxado.
TEST-27  Versão de política de R19 e definição dos períodos são imutáveis entre braços.
TEST-28  Gráficos brutos e tabelas têm hash de entrada rastreável.
```

**Gate crítico:** testes e smoke antes de campanha multi-seed. Logs imutáveis, código versionado, checksums e relatórios por seed devem acompanhar todo resultado.

---

## 12. ARTEFATOS E ESTRUTURA DAS ENTREGAS

```text
sqr_jc_decision/<protocol_version>/<run_id>/
  manifest.json                 # git sha, configs, units, hashes, versões
  stdout.log
  decisions.jsonl              # ou Parquet, schema versionado
  predictions.parquet
  kpis.parquet
  audit_5x3.csv
  r19_window_level.parquet
  resource_units_mapping.csv
  replay_index.parquet
  replay_actions.parquet
  replay_outcomes.parquet
  flags_r19.parquet
  metrics_per_seed.csv
  metrics_summary.csv
  figures/
  checksums.sha256
```

**Entrega textual obrigatória por etapa:**

- **D0:** mapa real de código, manifests, equivalência de baseline e replay; bloqueios.
- **D1:** frequência/magnitude da projeção; predições e violações estratificadas, com contagem de dados válidos.
- **D2:** tabela por estado `s_t`: `a_B0`, alternativas testadas, `a_recovery` quando houver, `A_common/A_relaxed`, métricas de SLA, evidência R19, hash do replay, classificação limitada à busca.
- **D3:** ações B0 × B1 decididas **antes** dos replays, métricas Rescue/Harm/Δ_allOK, OOD, decisão de avanço e justificativa.
- **GO_NO_GO.md:** matriz de evidências, limitações, problemas abertos, recomendação única (`GO-JOINT`, `STOP-JOINT`, `INCONCLUSIVE`) e se PILOT continua bloqueado.

### Modelo obrigatório para `GO_NO_GO.md`

```markdown
# GO / NO-GO — SQR Joint Selection

## Integridade — D0-A
- Resultado D0-A (PASS/STOP/INCONCLUSIVE):
- 5×3 linhas auditadas e sustentadas em traces:
- 5/5 revisado (ou outro resultado revalidado):
- mMTC a_m=0: evidence, denominator, status:
- R19 total/completas/não completas e denominadores:
- Unidade C e rastreio Python→MAC:
- Divergências de figuras e dados:
- Inventário de evidência ausente:

## Integridade — D0-B
- Commit e cenário:
- Hash dos manifests:
- D0-A/D0-B (PASS/STOP/INCONCLUSIVE):
- Replays A/A aprovados / total:
- Capacidade e evidência R19:

## D1 — seleção/projeção
- N decisões válidas:
- P(projeção) por carga/seed:
- Mudanças previstas após projeção:
- Associação com violação (não causal):

## D2 — oportunidade causal
- N estados elegíveis / testados / excluídos:
- N estados recuperáveis dentro de A_common:
- Seeds/episódios representados:
- N recuperáveis somente em A_relaxed:
- Cobertura da busca:
- Falhas/limitações:

## D3 — seletor automático
- Conjunto de teste independente:
- Δ_allOK pareado e incerteza:
- RescueRate / HarmRate / NetRescues:
- Resultados por slice e carga:
- OOD / tempo de decisão:
- Controles contra vazamento:

## Decisão
- [ ] GO-JOINT (autorizar pedido de PILOT)
- [ ] STOP-JOINT (não continuar esta hipótese)
- [ ] INCONCLUSIVE (indicar qual evidência falta)
- Justificativa objetiva:
- Afirmações que os dados AINDA NÃO permitem:
- Próximo passo permitido:
```

---

## 13. ORDEM DE EXECUÇÃO PARA O AGENTE

```text
[AGORA]
0. Ler documentos e identificar checkout atual.
1. Inventariar implementações, manifests, seeds, R19, contratos e replays.
2. Executar D0-A com revalidação dos cinco replays e R19; sem PASS bloquear.
3. Executar D0-B + gate de baseline, determinismo e MAC; sem PASS bloquear.
4. Reutilizar logs ou instrumentar SQR-HET; executar D1.
5. Pré-registrar amostragem, conjunto A_common e busca limitada.
6. Executar D2 em estados selecionados + controles A/A.
7. Se houver evidência de lacuna no mesmo espaço: construir B1 offline.
8. Testar D3 em estados/episódios reservados e com B0 pareado.
9. Emitir GO_NO_GO.md + relatórios/figuras/artefatos.
10. ENCERRAR. Não rodar PILOT/MAIN sem autorização explícita.

[SEMPRE]
- Preservar baseline e commits rastreáveis.
- Anotar toda premissa inferida como ASSUMIDA, nunca ARTIGO.
- Não inventar dados, parâmetros ou sucesso.
- Escalar bloqueios reais com diagnóstico preciso; não modificar metas para aprovar gates.
```

### Sequência de comandos de inspeção (genérica; executar na raiz real do repo)

```bash
# INSPEÇÃO, não executa simulação nem altera arquivos:
git status --short
git rev-parse HEAD
find MODA/docs/sqr -type f | sort | grep -E 'manifest|R19|contract|assumption|resource' || true
find MODA/ns-o-ran-gym -type f | grep -E 'sqr|replay' | head -100
find MODA/ns-3-dev/scratch/sqr -maxdepth 2 -type f | sort | head -100
```

**Não existem comandos universais garantidos para executar D1/D2/D3 até o inventário identificar os runners atuais.** Após D0, documentar no `INVENTORY.md` os comandos exatos de execução do checkout, seus argumentos, o caminho do ambiente Python, a versão do wrapper ns-3 e exemplos de saída. Não apresentar scripts ainda não criados como comandos já operacionais.

---

## 14. CRITÉRIOS DE NÃO-ALUCINAÇÃO E CONTROLE DE AFIRMAÇÕES

**Pode afirmar agora, como resultados REPORTADOS e ainda sujeitos a D0-A:** a caracterização encontrou violações com rádio saudável; cinco replays teriam mostrado recuperação ao usar uma ação fora de pisos conservadores; o hard classifier G4-B não passou seu gate; seleção conjunta é uma hipótese plausível e ainda não validada.

**Só após D0-A PASS:** os cinco replays e os contadores R19 anteriores foram revalidados em logs brutos, com evidência completa e unidade de recursos identificada.

**Só após D2:** há estados **identificados e reproduzidos** em que o SQR-HET deixa de aproveitar uma recuperação **dentro do mesmo espaço admissível**; isso deve ser qualificado pelo horizonte, carga, número de ações e cobertura testada.

**Só após D3:** os modelos atuais do SQR possuem sinal preditivo suficiente para que um seletor conjunto aproveite parte dessas oportunidades em estados congelados não vistos, sem acesso ao oracle; isso **não** garante benefício em malha fechada.

**Só após D4/D5 com resultados positivos:** a seleção conjunta melhora o desempenho de controle sob os perfis e condições experimentais avaliados, sem pioras materiais dos demais SLAs, com análise estatística adequada.

**Nunca afirmar sem teste apropriado:** ótimo global, inviabilidade física universal, ausência de violações em rede real, garantia probabilística de SLA derivada automaticamente das previsões, originalidade da representação de estado global, ou redução de violações já demonstrada para SQR-JC.

---

## 15. FONTES E HIERARQUIA DE AUTORIDADE

1. **Contexto de pesquisa entregue pela pesquisadora:** `contexto_novo_chat_sqr_jc_contribuicao(1).md` (estado consolidado até 06/11/2026; em especial §§ 6–10, 12, 14–16, 18–27, 32–33).
2. **Artigo-base entregue:** `Safe_RAN_Slicing_in_O-RAN_Minimizing_SLA_Violations_via_Model-Based_Reinforcement_Learning(7).pdf`, IEEE CCNC 2026, especialmente §§ III–V e Algoritmo 1.
3. **Checkout atual do projeto:** `AGENTS.md`, `MODA/docs/sqr/`, manifests, código SQR-HET e replays **verificados localmente**. Repositório público: https://github.com/Eliothluy/Jussi_SBRC . Documentação pública pode não conter as rodadas mais recentes.
4. **Parecer de auditoria fornecido:** `pesquisaCientifica.txt` — contém observações a TESTAR e interpretações que NÃO são validações automáticas. As imagens originais e logs descritos nesse parecer não foram anexados como ativos independentes neste chat; localizá-los no checkout para verificar.
5. **Consulta técnica auxiliar:** documentação do módulo NR 5G-LENA, https://cttc-lena.gitlab.io/nr/html/ . Confirme a API e o comportamento do scheduler **da versão efetivamente instalada**, sem importar pressupostos de outra release.

**Princípio final:** o experimento deve poder resultar honestamente em `STOP-JOINT`. O objetivo é identificar a contribuição cientificamente defensável, não construir um pipeline que obrigatoriamente aprove a coordenação conjunta.


## 16. BRIEFING PARA COLAR NO AGENTE EXECUTOR

> Este protocolo é a única fonte de passos para a campanha de decisão neste estágio. Comece pela etapa **D0-A** e não por D1/D2. Localize no checkout os dados brutos dos cinco estados × três ações, figure scripts, manifests de SLA e recursos, R19, checkpoint de replay e código do SQR-HET. Construa a matriz 5×3 dos resultados eMBB/URLLC/mMTC com `PASS/FAIL/UNKNOWN` e razão explicada para cada ação, sobretudo mMTC quando `a_m=0`. Recalcule R19 por run/seed/λ, reconciliando 24.482/12.075/12.407 sem escolher números de gráficos. Descubra se C=53 é RB/PRB/RBG ou unidade lógica e rastreie efetiva aplicação no MAC. Reproduza A/A dos cinco estados e confira coortes/rolling URLLC. Registre falhas sem alterá-las silenciosamente. **Pare e entregue D0A_GATE.json e relatório auditável.** Só depois de `D0A_PASS` prossiga para D0-B e, após seu gate, D1→D2→D3. Se não existirem dados brutos, classifique como INCONCLUSIVE; não gere dados artificiais. Em D2, avalie recuperações em **A_common idêntico** para B0/B1; as recuperações que dependem da remoção de pisos não sustentam, sozinhas, Joint Selection. Em D3, mantenha modelos SQR congelados e o replay/oracle fora da seleção. **Nunca executar PILOT/MAIN, mudar SLA/TDD/kernel ou declarar SQR-JC superior sem autorização e evidências.**

### Checklist final de entrega desta rodada

- [ ] Inventário/hashes dos dados e scripts originais.
- [ ] Tabela `D0A_replay_5x3.csv` com evidência por slice, não inferida do PNG.
- [ ] Verificação de `(37,13,3)` nas três janelas citadas: qual slice falha?
- [ ] Verificação de `(48,5,0)`: mMTC com zero unidades e elegibilidade do SLA.
- [ ] R19 refeito com contagem por run/seed/λ e denominador explícito.
- [ ] Resource mapping rastreado Python→IPC→C++→scheduler/MAC.
- [ ] Replay A/A e diferenças de prefixo/exógenos registrados.
- [ ] URLLC coorte pós-ação distinguida de rolling de 1 s.
- [ ] `D0A_GATE.json` emitido e decidido sem maquiagem.
- [ ] D0-B/D1/D2/D3 executados apenas se gates anteriores passarem.
- [ ] D4/D5 não foram iniciados.
- [ ] `GO_NO_GO.md` preserva a possibilidade de `STOP-JOINT` ou `INCONCLUSIVE`.
