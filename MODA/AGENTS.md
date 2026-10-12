# AGENTS.md

Contexto vigente para este workspace: considerar este arquivo como a referência principal para qualquer agente de IA que trabalhe no paper, na revisão textual, na análise metodológica, na implementação experimental ou na organização dos resultados.

Este documento deve preservar o escopo atual da pesquisa. Agentes não devem reformular o problema, trocar as variáveis de decisão, substituir o objetivo experimental ou alterar o conjunto principal de comparações sem instrução explícita do autor.

---

## 1. Contexto do Paper

O paper investiga como diferentes meta-heurísticas se comportam na otimização da distribuição de recursos entre slices em redes 5G/O-RAN.

O problema é formulado como um problema de otimização contínua, no qual os pesos de alocação dos slices são as variáveis de decisão. A formulação atual deve ser mantida durante a escrita, revisão e implementação.

O objetivo experimental central é avaliar a capacidade de diferentes métodos de busca em encontrar bons vetores de pesos de alocação para os slices. A conclusão principal do trabalho deve indicar, de forma explícita, qual meta-heurística encontrou as melhores soluções para essa formulação.

---

## 2. Escopo Inalterável da Pesquisa

Os seguintes pontos são decisões de projeto e devem ser preservados:

- O problema continua sendo tratado como otimização contínua.
- As variáveis de decisão continuam sendo os pesos de alocação dos slices.
- A formulação atual do problema deve ser mantida.
- As meta-heurísticas principais avaliadas são PSO, GA, SA e a meta-heurística híbrida reformulada pelo trabalho.
- A comparação deve permanecer centrada na qualidade das soluções encontradas para a formulação com pesos de alocação como variáveis de decisão.
- AQPS e RSLAQ devem ser mantidos como bases conceituais e comparativas do paper.
- Os schedulers puros RR, BCQI e PF devem ser usados como baselines de comparação.

Agentes não devem deslocar o foco do artigo para aprendizado por reforço, deep learning, projeto de novos schedulers, otimização discreta, programação inteira ou outra formulação alternativa, salvo se o autor solicitar explicitamente.

---

## 3. Formulação do Problema

A formulação deve ser apresentada como um problema de otimização contínua aplicado à distribuição de recursos entre slices em redes 5G/O-RAN.

### 3.1 Variáveis de decisão

A solução candidata deve ser representada por um vetor de pesos de alocação:

\[
\mathbf{w} = [w_1, w_2, \ldots, w_S]
\]

em que:

- \(S\) é o número de slices considerados no cenário experimental;
- \(w_i\) representa o peso de alocação associado ao slice \(i\);
- \(\mathbf{w}\) é a entrada otimizada pelas meta-heurísticas.

A interpretação dos pesos deve permanecer alinhada à formulação original do paper. Caso a implementação utilize normalização, projeção ou penalização para garantir viabilidade, isso deve ser descrito explicitamente na metodologia.

### 3.2 Domínio das variáveis

Salvo definição diferente já estabelecida no paper ou no código experimental, os pesos devem respeitar as seguintes condições gerais:

\[
w_i \geq 0, \quad \forall i \in \{1, \ldots, S\}
\]

\[
\sum_{i=1}^{S} w_i = 1
\]

Se a formulação oficial usar limites inferiores, limites superiores, normalização alternativa ou pesos não necessariamente normalizados, a versão oficial da formulação deve prevalecer. O agente deve apenas documentar essa escolha, não substituí-la.

### 3.3 Função objetivo

A função objetivo deve avaliar a qualidade de um vetor de pesos \(\mathbf{w}\) com base nas métricas de desempenho definidas no paper.

A estrutura geral deve ser descrita como:

\[
\max_{\mathbf{w}} F(\mathbf{w})
\]

ou, de forma equivalente, como minimização:

\[
\min_{\mathbf{w}} J(\mathbf{w}) = -F(\mathbf{w})
\]

A escolha entre maximização e minimização deve seguir a implementação experimental. Agentes não devem inverter o sentido da função objetivo sem verificar a convenção usada nos experimentos.

A definição de “melhor solução” deve estar diretamente associada ao valor da função objetivo, respeitando restrições de viabilidade e critérios experimentais definidos.

### 3.4 Restrições e penalidades

Caso uma solução candidata viole restrições da formulação, a metodologia deve explicar como a violação é tratada. Estratégias aceitáveis incluem:

- normalização do vetor de pesos;
- projeção para o domínio viável;
- penalização na função objetivo;
- descarte ou reparo da solução inviável.

A estratégia escolhida deve ser aplicada de forma consistente a PSO, GA, SA e à meta-heurística híbrida para evitar viés experimental.

---

## 4. Métodos Avaliados

### 4.1 Meta-heurísticas principais

O estudo deve comparar as seguintes meta-heurísticas:

- PSO — Particle Swarm Optimization;
- GA — Genetic Algorithm;
- SA — Simulated Annealing;
- Meta-heurística híbrida reformulada pelo trabalho.

A descrição de cada método deve enfatizar seu papel na busca por vetores de pesos de alocação. A explicação não deve transformar os algoritmos em schedulers independentes; eles atuam como mecanismos de otimização dos pesos.

### 4.2 Meta-heurística híbrida

A meta-heurística híbrida deve ser apresentada como a contribuição reformulada do trabalho. Sua descrição deve deixar claro:

- quais componentes algorítmicos são combinados;
- qual etapa favorece exploração global;
- qual etapa favorece exploração local ou refinamento;
- como uma solução candidata é representada;
- como soluções são atualizadas;
- como a função objetivo é avaliada;
- qual critério de parada é usado;
- por que a hibridização é adequada para a formulação contínua com pesos de alocação.

Agentes não devem inventar uma arquitetura híbrida diferente da usada no paper. Se a definição ainda estiver incompleta, o texto deve sinalizar a necessidade de explicitar a estrutura da híbrida, em vez de substituir a proposta.

---

## 5. Baselines e Comparações Obrigatórias

### 5.1 Comparação entre meta-heurísticas

A comparação principal deve ser feita entre:

- PSO;
- GA;
- SA;
- meta-heurística híbrida.

O objetivo dessa comparação é identificar qual método encontra as melhores soluções para a formulação proposta.

### 5.2 Comparação com schedulers puros

Os resultados das heurísticas e meta-heurísticas devem ser comparados com schedulers puros:

- RR — Round Robin;
- BCQI — Best Channel Quality Indicator;
- PF — Proportional Fair.

Essa comparação deve mostrar se a otimização dos pesos de alocação gera melhoria em relação ao uso direto de schedulers tradicionais.

### 5.3 Comparação com RSLAQ e AQPS

O paper deve manter AQPS e RSLAQ como bases relevantes.

A comparação com RSLAQ deve ser mantida na análise experimental ou na discussão dos resultados, conforme a estrutura atual do paper. RSLAQ representa a base orientada por aprendizado/reinforcement learning usada para posicionar a proposta em relação a métodos adaptativos existentes; quando os resultados da Phase 4 estiverem disponíveis, eles devem entrar como baseline DRL principal do paper.

AQPS deve continuar sendo usado como base conceitual, metodológica ou comparativa, conforme definido na pesquisa. No estado atual do código experimental, AQPS aparece como o modo `slice_aqps`: um baseline slice-aware que calcula orçamentos inteiros de RBGs por slice com garantia mínima para demandas ativas e ajuste por urgência/prioridade. Ele deve ser tratado como baseline adaptativo de alocação por slice, não como substituto da formulação principal baseada na otimização contínua dos pesos.

RSLAQ e AQPS servem para responder se a otimização offline por meta-heurísticas é competitiva frente a abordagens adaptativas já consideradas no trabalho. A redação deve preservar essa função comparativa sem deslocar o foco do artigo para DRL, aprendizado por reforço ou projeto de novos schedulers.

Agentes devem evitar remover RSLAQ ou AQPS da narrativa do paper.

---

## 6. Protocolo Experimental Recomendado

A metodologia experimental deve ser suficientemente detalhada para permitir reprodutibilidade.

Sempre que os dados estiverem disponíveis, o paper deve reportar:

- número de execuções independentes por algoritmo;
- número máximo de iterações ou avaliações da função objetivo;
- tamanho da população ou enxame, quando aplicável;
- parâmetros específicos de PSO, GA, SA e da híbrida;
- critérios de parada;
- sementes aleatórias ou procedimento de controle de aleatoriedade;
- ambiente de simulação;
- configuração dos slices;
- métricas de rede usadas na função objetivo;
- tempo de execução ou custo computacional, quando relevante.

Como recomendação estatística, quando não houver restrição contrária, usar múltiplas execuções independentes para cada método. Um valor típico é pelo menos 30 execuções por algoritmo, desde que o custo computacional permita.

---

## 7. Métricas de Avaliação

A comparação deve priorizar a qualidade das soluções encontradas para a formulação proposta.

Sempre que possível, reportar:

- melhor valor da função objetivo;
- média dos valores obtidos;
- mediana;
- desvio padrão;
- intervalo interquartil;
- pior valor obtido;
- curva de convergência;
- número de avaliações da função objetivo;
- tempo de execução;
- taxa de soluções viáveis, se aplicável.

Métricas de rede, como vazão, latência, justiça, perda, eficiência espectral ou satisfação de requisitos por slice, devem ser incluídas apenas se fizerem parte da formulação ou da avaliação experimental do paper.

---

## 8. Análise Estatística

A conclusão sobre a melhor meta-heurística deve ser sustentada por evidências quantitativas.

Quando houver múltiplas execuções independentes, recomenda-se aplicar análise estatística apropriada, como:

- teste de Friedman para comparação global entre múltiplos algoritmos;
- pós-teste de Nemenyi, Holm ou Bonferroni quando necessário;
- teste de Wilcoxon pareado para comparações dois a dois;
- teste de Kruskal-Wallis quando a estrutura dos dados justificar comparação não paramétrica independente.

O paper deve evitar afirmar superioridade absoluta de um método com base em apenas uma execução ou em diferenças pequenas sem análise de variabilidade.

Caso testes estatísticos não sejam usados, a limitação deve ser reconhecida e a conclusão deve ser formulada com cautela.

---

## 9. Regras de Escrita e Interpretação

Ao escrever ou revisar o paper, agentes devem seguir estas regras:

- Usar linguagem técnica compatível com redes 5G/O-RAN, network slicing e otimização por meta-heurísticas.
- Manter o foco na otimização dos pesos de alocação dos slices.
- Explicar que PSO, GA, SA e a híbrida procuram soluções candidatas no espaço contínuo dos pesos.
- Diferenciar claramente meta-heurísticas de otimização e schedulers puros.
- Não afirmar que a híbrida é melhor sem sustentação nos resultados experimentais.
- Não remover comparações com RR, BCQI, PF, RSLAQ ou AQPS.
- Não alterar a formulação matemática sem autorização explícita.
- Não introduzir métodos adicionais como se fossem parte do objetivo central do paper.
- Evitar conclusões vagas; a conclusão deve dizer objetivamente qual método apresentou o melhor desempenho segundo os critérios experimentais.

---

## 10. Estrutura Recomendada para a Discussão dos Resultados

A discussão dos resultados deve seguir uma linha argumentativa clara:

1. Apresentar o desempenho dos schedulers puros RR, BCQI e PF.
2. Apresentar o desempenho das meta-heurísticas PSO, GA, SA e híbrida.
3. Comparar as meta-heurísticas entre si com base na função objetivo e nas métricas de rede relevantes.
4. Comparar os resultados obtidos com RSLAQ e, quando aplicável, com AQPS.
5. Discutir convergência, estabilidade e custo computacional.
6. Indicar qual método encontrou as melhores soluções para a formulação adotada.
7. Reconhecer limitações experimentais, se houver.

---

## 11. Conclusão Esperada do Paper

A conclusão do paper deve ser objetiva e diretamente conectada ao objetivo experimental.

Ela deve responder explicitamente:

- qual meta-heurística encontrou as melhores soluções;
- em quais métricas essa superioridade foi observada;
- se o ganho em relação a PSO, GA e SA foi consistente;
- se houve vantagem em relação aos schedulers puros RR, BCQI e PF;
- como os resultados se posicionam em relação ao RSLAQ e às bases do AQPS;
- quais limitações permanecem para trabalhos futuros.

A conclusão não deve ser genérica. Expressões vagas como “os métodos apresentaram bons resultados” devem ser acompanhadas de evidência quantitativa e indicação clara do método com melhor desempenho.

---

## 12. Checklist para Agentes

Antes de finalizar qualquer contribuição para o paper, o agente deve verificar:

- A formulação continua sendo de otimização contínua?
- Os pesos de alocação dos slices continuam sendo as variáveis de decisão?
- PSO, GA, SA e a híbrida foram mantidos como métodos principais?
- RR, BCQI e PF foram mantidos como schedulers puros de comparação?
- RSLAQ e AQPS continuam presentes na narrativa do paper?
- A comparação está centrada na qualidade das soluções encontradas?
- A metodologia informa parâmetros, critérios de parada e número de execuções?
- A análise apresenta média, dispersão e melhor solução quando possível?
- A conclusão aponta objetivamente o método vencedor?
- O texto evita afirmações de superioridade sem suporte experimental?

---

## 13. Restrições para Agentes

Agentes não devem:

- trocar a formulação contínua por formulação discreta;
- trocar os pesos dos slices por outra variável de decisão;
- remover PSO, GA, SA ou a híbrida do conjunto principal;
- substituir RR, BCQI e PF por outros schedulers sem autorização;
- remover RSLAQ ou AQPS da comparação;
- criar resultados numéricos inexistentes;
- inventar parâmetros experimentais se eles não estiverem definidos;
- afirmar significância estatística sem teste apropriado;
- declarar que a híbrida é superior antes de verificar os resultados;
- transformar o artigo em uma proposta de novo scheduler em vez de uma análise de meta-heurísticas para otimização dos pesos.

Quando alguma informação estiver ausente, o agente deve sinalizar a lacuna e propor uma forma de preenchê-la sem alterar o escopo do paper.

---

## 14. Contexto Operacional e Estado Experimental (Sessões 2026-07-07 a 2026-07-14)

Esta seção registra o estado experimental vigente, as mudanças aplicadas e os bugs conhecidos, de forma que agentes futuros herdem o contexto sem precisar rederivá-lo. Ela complementa (não substitui) as seções anteriores; em caso de conflito sobre escopo ou formulação, prevalecem as seções 1-13.

### 14.1 Campanha vigente

- **Run tag**: `20260626_122326`, em `ns-o-ran-gym/results_controlled/heuristics_metaheuristics/20260626_122326/`.
- **Decisão de projeto**: ambiente controlado com **3 seeds (1, 2, 3)** e `RUNS=1`. Não há intenção de expandir para 30 execuções por restrição de tempo de simulação. Agents não devem introduzir 30 seeds por iniciativa própria.
- **Hiperparâmetros vigentes**: `META_ITERATIONS=12`, `META_POPULATION=6`, `META_RANDOM_SEED=2026`, `META_MUTATION_STRENGTH=0.12`, `META_INTRA_ALGO=PF`. Cada par (cenário, seed) executa GA (72 evals), PSO (72), SA (72, orçamento equalizado a `iterations×population`) e híbrida (84 = `iterations×(population+1)`, o +1 é o refinamento local SA por iteração), totalizando **300 avaliações da função objetivo por par**. O orçamento de avaliações do SA foi equalizado ao do GA/PSO (72) para uma comparação justa sob igual custo computacional; anteriormente o SA fazia apenas 12 evals (busca por trajetória única), o que era uma comparação injusta.
- **Fases**: Phase 1 (baselines) está 210/211 OK; Phase 2 (meta-heurísticas) segue em execução; Phase 3 (meta-eval) e Phase 4 (RSLAQ DDQN paper-faithful) estão agendadas na sequência via `resume_campaign.sh`. Na auditoria de 2026-07-14, 11 dos 15 pares cenário×seed estavam completos com GA=72, PSO=72, SA=72 e híbrida=84. `stressed/seed=3` estava parcial (GA=72, PSO=72, SA=52, híbrida=0) e `insufficient_resources` ainda estava no GA (seeds 1/2/3 com 57/48/27 avaliações, respectivamente). Esses números são um snapshot e devem ser recontados antes de qualquer consolidação final.
- **AQPS na campanha**: o baseline AQPS está habilitado como `slice_aqps` nos scripts de baselines e aparece no manifesto `heuristics_ns3/results_rslaq_network_only/batch_manifest.csv`. No estado verificado desta sessão, há 15 execuções planejadas de `slice_aqps`: 14 `ok` com `summary.csv` e 1 `run_failed` em `congestion`, `seed=1`, `run=1` (`ns3_run_failed`).

### 14.2 Mudanças de código aplicadas nesta sessão (não descrever como bugs novos)

Estas alterações já estão aplicadas e devem ser tratadas como o estado corrente:

1. **Checkpoint de evals em `run_rslaq_metaheuristics.py`**: cada avaliação grava um sidecar `candidate.json` no diretório do eval. Ao reiniciar, o closure `evaluate` desserializa e pula o ns-3 quando os pesos pedidos batem com o cache (tolerância 1e-9). A persistência do `metaheuristic_results_*.csv` e do `best_candidate_*.json` é incremental (após cada método). O `import csv` foi adicionado (bug bloqueante que impedia a Phase 2 de salvar resultados).
2. **Calibração da penalidade URLLC em `scoring.py`**: o divisor `(delay - 10)/200` foi trocado por `/20`. Com o divisor antigo, a penalidade era irrisória na faixa operacional (14,864 ms → −0,6 pts), permitindo que o "ótimo" violasse o SLA de 10 ms. Agora: 14,864 ms → −6,08 pts; saturação em 30 ms (−25 pts). O scoring usa `delay_ms_mean` (ponderado por pacote); os percentis `delay_ms_p95/p99` **não são usados** na função objetivo.
3. **Re-score do cache**: o script `examples/rescore_cache.py` recalcula scores dos `summary.csv` em cache quando o `scoring.py` muda, sem re-rodar o ns-3. Os 150 evals em cache foram re-scored após a mudança acima (66 mudaram, delta médio −19,4 pts).
4. **Hiperparâmetros**: `META_ITERATIONS` 4→12 em `run_all_scenarios.sh` e `resume_campaign.sh`. O cache é reaproveitado porque o RNG é seeded e os pesos dos primeiros evals são deterministicamente reproduzíveis.

### 14.3 Bugs e limitações conhecidas (documentar no paper, não corrigir por iniciativa própria)

- **Percentis de delay amostrados por janela, não por pacote**: no `rslaq-sim.cc`, `delay_ms_p95/p99` são calculados sobre médias por janela de 10 ms, o que produz inconsistências do tipo `delay_ms_mean > delay_ms_p95` em ~20% das linhas. A correção exigiria modificar o `StatsCallback` no C++ e re-rodar toda a campanha (Phase 1+2). **Decisão: adiar e documentar como limitação** (o scoring usa `delay_ms_mean`, então a otimização não é afetada).
- **SA com escala de aceitação ad-hoc**: o orçamento do SA foi equalizado a 72 evals (= GA/PSO), resolvendo a inequidade anterior (12 evals). A escala da aceitação de Metropolis (`temperature * 20.0`) permanece uma compensação ad-hoc para a escala da função objetivo.
- **PSO/Híbrida com renormalização pós-passo**: a projeção para o simplexo após cada atualização de velocidade destrói a semântica canônica do vetor velocidade. Funciona empiricamente mas não é PSO estritamente canônico.
- **Hiperparâmetros ainda modestos**: `population=6, iterations=12` está acima do regime "piloto" anterior mas abaixo do ideal para meta-heurísticas. Aumentar a população invalidaria o cache existente.
- **Orçamento da híbrida ainda superior**: GA, PSO e SA têm 72 avaliações cada, mas a híbrida tem 84, isto é, 16,7% a mais devido ao refinamento local adicional. A comparação primária deve usar o teto comum de 72 avaliações para os quatro métodos; as avaliações 73-84 da híbrida podem ser apresentadas separadamente como efeito adicional do refinamento local, sem necessidade de reexecutar o ns-3. Nos 11 pares completos auditados em 2026-07-14, as avaliações extras melhoraram o melhor score da híbrida em dois pares e mudaram o vencedor em um (`stressed`, seed 1).
- **Inicialização não equivalente**: GA e PSO começam com vetores aleatórios normalizados; SA começa na vizinhança de `[1/3,1/3,1/3]`, avaliando primeiro uma mutação; a híbrida usa `seed_weight_candidates`. Embora existam oito priors no código, com `population=6` somente os seis primeiros são usados, sem candidatos aleatórios na população inicial. A inicialização guiada é parte do método, mas impede atribuir eventual ganho exclusivamente aos operadores GA/PSO/SA sem estudo de ablação.
- **RNG compartilhado em ordem fixa**: para cada par cenário×seed, um único `random.Random(2026)` é recriado e consumido sequencialmente por GA, PSO, SA e híbrida. Os métodos não recebem streams independentes por algoritmo. Essa decisão deve ser documentada e considerada ao interpretar diferenças de inicialização e trajetória.

### 14.4 Achados da revisão crítica (devem orientar a redação, não serem revertidos)

Uma revisão crítica de pesquisa identificou pontos que devem ser refletidos na redação do paper:

- **Os nomes dos cenários são perfis nominais de entrada, não estados operacionais garantidos**. As cargas úteis totais vigentes são 57 Mbps (`low_traffic`), 73 Mbps (`normal`), 128 Mbps (`stressed`), 245 Mbps (`congestion`) e 295 Mbps (`insufficient_resources`). `congestion` apresenta congestionamento inequívoco; `stressed` é defensável como regime degradado; `insufficient_resources` significa recursos insuficientes para toda a demanda oferecida, mas não foi demonstrado como inviável frente aos SLAs mínimos de throughput. `low_traffic` e `normal` podem degradar fortemente em algumas seeds por efeito de geometria/canal e não devem ser descritos como necessariamente descongestionados.
- **A separação entre schedulers é pequena ou moderada fora do congestionamento**, e grande parte do throughput em `low_traffic`, `normal` e `stressed` é determinada pela seed. Esse resultado negativo deve ser apresentado honestamente, mas sem classificar automaticamente os três cenários como não saturados.
- **Topologia e carga estão confundidas**: todos os cenários usam uma única gNB com painel direcional de 65°×7°, bearing 0°, tilt 6°, enquanto os UEs são distribuídos em um disco completo de 360° e raio 100 m; não há configuração explícita de beamforming helper no `rslaq-sim.cc`. Além disso, cada cenário muda simultaneamente número de UEs, composição por slice, posições amostradas, taxa e tamanho de pacote. A alta variabilidade entre seeds pode refletir cobertura/MCS, não apenas disputa por recursos.
- **Métricas de uso exigem interpretação cuidadosa**: `budget_utilization_pct=100` mede uso do orçamento concedido ao slice, não ocupação física total da célula; o throughput do FlowMonitor inclui cabeçalhos IP/UDP e pode exceder a carga útil configurada, especialmente com pacotes URLLC de 50 bytes; `pdr_pct` pode ser baixo enquanto `plr_pct` permanece próximo de zero porque pacotes ainda em fila no final não são necessariamente classificados como perdidos. Para diagnosticar congestionamento, priorizar PDR, atraso médio, filas e RX/TX, não `plr_pct` isoladamente.
- **A híbrida perde para o PSO puro em 2 dos 3 cenários** relatados no draft atual. A contribuição não deve ser posicionada como "híbrida vencedora" sem evidência robusta. Posicionamento recomendado: híbrida como avaliação comparativa, com foco metodológico em **META_RISK_ELASTIC** (blending prior×online) e **AQPS** (budgets inteiros com garantia mínima) como baselines mais sólidas.
- **N=3 seeds é insuficiente para testes estatísticos formais** (Wilcoxon/Friedman exigem n≥5). O paper deve reportar média±desvio e melhor/mediana, **não declarar significância**, e reconhecer a limitação explicitamente, remetendo significância formal a trabalho futuro.
- **RSLAQ DDQN (Phase 4) é o baseline DRL do paper** e deve entrar na comparação principal quando disponível. Ele responde à pergunta "por que meta-heurísticas e não DRL?" que um revisor fará.

### 14.5 Diretrizes operacionais para agents

- **Não commitar** `results_controlled/`, `*.pid`, logs de campanha (`resume_campaign.log`, `.meta_*.log`, `run_*_parallel*.log`). O `.gitignore` da raiz e o `ns-o-ran-gym/.gitignore` já os protegem.
- **Ao mudar `scoring.py`**, executar `examples/rescore_cache.py` para atualizar os sidecars em cache antes de relançar a Phase 2.
- **Ao mudar hiperparâmetros que afetem o RNG** (population, random_seed), o cache é invalidado e tudo precisa re-rodar do zero. Mudar apenas `iterations` preserva o cache.
- **Para parar a campanha**: SIGTERM ao PID em `resume_campaign.pid` pode não propagar aos filhos; usar `pkill -KILL -f "ns3.46-rslaq-sim-default|run_rslaq_metaheuristics.py|run_all_scenarios.sh"` se necessário. O cache (sidecars já escritos) sobrevive a SIGKILL.
- **Relevância**: o SOTA em RAN slicing (2024-2025) é dominado por DRL (DDPG, PPO, federated/hierarchical RL). O paper deve articular explicitamente por que meta-heurísticas offline (auditáveis, estáveis, sem treinamento) são a escolha certa — caso contrário, um revisor questionará por que não DRL.

### 14.7 Campanha v2 — reformulação sancionada da auditoria (2026-07-17)

A auditoria (`auditoria_tecnica_completa.md`, verificada contra os dados brutos: ~95% fiel) motivou uma **reformulação sancionada pelo usuário**. Isto NÃO revoga as limitações documentadas da campanha **v1** (§14.3/§14.4): a v1 permanece intacta (snapshot `resultados_cenarios_finalizados_20260715/` + backup `~/Documentos/artigo_jussi_backup_20260717/`) e suas limitações continuam válidas para o dataset v1. A v2 é uma linha **separada e versionada**; nunca misturar métricas/scores v1 e v2 sem rótulo.

**Correções aplicadas (Fases 1–2):**
- **Percentis de delay por pacote** (corrige §14.3): `rslaq-sim.cc` agora agrega o `delayHistogram` do FlowMonitor por slice (`DelayBinWidth=0.5ms`); `delay_ms_p95/p99` viram por-pacote e há nova coluna `delay_ms_p999`. `mean > p95` deixa de ser artefato (pode ainda ocorrer legitimamente em cauda pesada).
- **Novas colunas** no `summary.csv`: `delay_ms_p999`, `deadline_violation_pct`, `reliability_in_time_pct`, `sla_thr_pct`, `sla_pdr_pct`, `sla_delay_pct`, `plr_detected_pct`.
- **`sla_satisfaction_pct` composto**: `min(S_thr, S_pdr, S_delay)` com alvos por slice em `SLA_TARGETS` (elimina o paradoxo SLA=100% com PDR baixo/delay alto). O antigo throughput-only foi substituído.
- **PLR reconciliado**: `plr_pct = (tx-rx)/tx` (perda efetiva, inclui filas residuais); a métrica antiga do FlowMonitor vira `plr_detected_pct`.
- **HARQ**: `LogHarqState` implementado (corrige o `harq_tracking.csv` vazio); campos de processo HARQ são `NA` (não expostos nesta subclasse) — sem fabricação.
- **Score v2** (`scoring.py::score_summary_rows_v2`): feasibility-first **recalibrado no piloto (2026-07-18)**. O piloto congestion×10 seeds mostrou que, dos alvos de SLA, só a latência de cauda URLLC é fisicamente atingível sob sobrecarga (97/97 candidatos violavam PDR_URLLC≥90%, PDR_MTC≥80% e eMBB≥60% da oferta — tetos observados: 81%, 72% e 48%); com múltiplos gates, a região viável era vazia. **Design final (Path C):** única restrição RÍGIDA = p99_URLLC≤10ms (por pacote); PDR/throughput entram GRADUADOS via satisfação composta por slice; feasível → `100·(0.5·min_SLA + 0.5·mean_SLA)` (fairness max-min pune starvation); infeasível → score<0 proporcional ao excesso de latência. Validado nos 97 evals reais: 47% viáveis, score discrimina em [0,31]. `score_summary_rows` (v1) permanece intacta.
- **Runner/rescore/cache**: `--score_version {v1,v2}` e `--per_seed_search`; o sidecar grava `score_version` e o cache não mistura versões. `run_all_scenarios.sh`: env `META_SCORE_VERSION`, `META_PER_SEED_SEARCH`; `REPO_ROOT` agora derivado da localização do script (corrige o hardcode `/home/elioth`).

**Campanha v2 (Fase 3):** orçamento reduzido justificado pela convergência em ~10 evals (§10 da auditoria): `META_ITERATIONS=8 META_POPULATION=6` (48 evals GA/PSO/SA, 56 híbrida), **10 seeds**, output root `results_controlled/heuristics_metaheuristics_v2/`. Piloto: `congestion × 10 seeds` para validar scoring v2 e **calibrar** os limiares de viabilidade (o PDR≥99.999% do §D da auditoria é inatingível nesta simulação; usar limiares realistas que deixem ≥1 região do simplex viável por cenário). Com n≥5 os testes pareados (Wilcoxon/Friedman) tornam-se válidos — ver `examples/analyze_v2_stats.py`.

**Protocolo em dois níveis (decisão 2026-07-19):** `SIM_TIME=15` em congestion custa ~1.5–2h por eval, tornando a busca a 15s inviável (semanas). Adotado o protocolo padrão de otimização cara: **(nível 1) busca com fitness a `SIM_TIME=5`** (idêntico ao protocolo v1) e **(nível 2) validação de alta fidelidade a `SIM_TIME=15`** apenas dos best candidates por (cenário, seed). Declarar ambos os níveis no paper. Material do nível 2 já existente: `v2_pilot_congestion_15s_partial/` (29 evals GA a 15s, Path C, binário com fix do IpForward) e `v2_pilot_congestion_precalib/` (97 evals a 15s, scores pré-Path C — usar apenas os summary.csv, re-pontuando).

**Brecha de scoring Path C fechada (2026-07-23, revelada pelo piloto):** um URLLC
MORTO (0 pacotes entregues → histograma de delay vazio → `delay_ms_p99=NA`) era
tratado por `safe_float("NA")→0 ≤ 10ms` como "latência-viável", permitindo ao
otimizador satisfazer a restrição rígida MATANDO o URLLC (reintroduzindo o
paradoxo de starvation da v1). Corrigido em `score_summary_rows_v2`: p99 ausente/NA
OU PDR_URLLC≈0 → latência sentinela `V2_URLLC_DEAD_P99_MS=10000` → infeasível.
Impacto medido no piloto: 22/943 evals (2.3%) eram falsos-viáveis; corrige
best-por-seed de ≥1 seed. **Ação:** re-pontuar o cache do piloto com
`rescore_cache.py --score_version v2` ANTES da análise (recomputa dos summaries,
sem re-rodar ns-3); os baselines já usarão o fix. Teste: `test_scoring_v2.py::test_dead_urllc_is_infeasible_not_feasible`.

**Comparação central da v2 = otimização vs NÃO-otimização (decisão 2026-07-21):** o claim do paper não é "qual metaheurística vence" nem comparar cenários entre si — é que a **otimização metaheurística supera baselines sem otimização**, sob Path C, no mesmo protocolo. Os baselines da v1 (`heuristics_ns3`) NÃO são reaproveitáveis (binário antigo, sem percentil por-pacote nem `sla_satisfaction_pct` composto) — precisam ser RE-EXECUTADOS com o binário v2. Baselines "sem otimização": schedulers puros (pure_rr/pf/bcqi), pesos fixos (psta_equal, slice_weighted_*), heurísticas adaptativas (slice_aqps, slice_*_greedy, ...). Cada um é **1 run** (não uma busca), pontuado com `score_summary_rows_v2` (Path C), comparado ao best candidate metaheurístico por (cenário, seed). **Ordem:** rodar APÓS o piloto metaheurístico (sem contenção de CPU). Comando (congestion × 10 seeds, 5s, binário v2):
```
cd ns-o-ran-gym
REPO_ROOT=/home/eliothluy/Documentos/artigo_jussi RUN_TAG=v2_baselines_congestion \
OUTPUT_ROOT=$PWD/results_controlled/heuristics_metaheuristics_v2/v2_baselines_congestion \
SCENARIOS="congestion" SEEDS="1 2 3 4 5 6 7 8 9 10" SIM_TIME=5 \
RUN_BASELINES=1 RUN_METAHEURISTICS=0 RUN_META_EVALUATION=0 RUN_RSLAQ_DDQN_PAPER=0 \
PARALLEL_JOBS=10 bash examples/run_all_scenarios.sh
```
Depois: estender `analyze_v2_stats.py` para pontuar os `summary.csv` dos baselines com Path C e tabelar best-metaheurística vs cada baseline por seed (Wilcoxon pareado, n=10). Nota: schedulers puros não têm vetor de pesos, mas produzem KPIs por slice → pontuáveis por Path C normalmente.

**Modelo de custo CORRETO desta máquina (bisseção 2026-07-19 — não repetir o erro):** congestion com pesos balanceados custa **~5–7 min de wall-clock por segundo simulado** (~25–35 min por eval a 5s; ~1.5–2h a 15s), e SEMPRE custou — verificado com bancada de 4 variantes (código v2 atual, código pré-v2, binwidth 1ms, HARQ-log off: todas idênticas, 750ms simulados em 300s). As métricas v2 e o fix do IpForward NÃO tornaram o binário mais lento. Os "47–160s por eval" da auditoria (§2) foram medidos na MÁQUINA ANTIGA (`elioth`), ~15× mais rápida neste regime — não usar esses números para planejar campanhas aqui. Cenários leves (low_traffic/normal) são ~5–10× mais baratos por eval. Piloto congestion×10 seeds a 5s com 48 evals/método: ~2 dias com `PARALLEL_JOBS=10`.

**Custo por cenário MEDIDO (probe 5s, 2026-07-25):** low_traffic <1 min/eval (completa 5s em <180s); normal ~5 min/eval (2980ms em 180s); stressed ~16 min/eval (920ms em 180s); insufficient_resources ~27 min/eval (550ms em 180s, o mais pesado, ~= congestion); congestion ~30-55 min/eval. Ordenar campanhas do mais barato ao mais caro para resultados incrementais.

**Estado da campanha v2 (2026-07-25):**
- `v2_pilot_congestion/` — CONCLUÍDO: metaheurísticas (999 evals, iter=4, Path C corrigido, re-scored) + 140 baselines. Resultado central: OTIMIZAÇÃO 7/10 seeds latência-viável vs ≤4/10 de qualquer baseline (PF/RR/AQPS = 0/10); heurísticas gulosas MATAM o URLLC (3-5% RBG → p99 ~1s). Metaheurísticas empatam entre si (Friedman p=0.29). Análise: `analyze_v2_opt_vs_baseline.py`, CSV `v2_opt_vs_baseline_per_slice.csv`.
- `v2_all_scenarios/` — RODANDO (~4-5 dias): low_traffic+normal+stressed+insufficient_resources, baselines (560) + metaheurísticas (~4000 evals), mesma config. Ao concluir, análise otimização-vs-baseline por-slice dos 5 cenários (combinar os dois roots).

**Análise v2:** `examples/analyze_v2_stats.py` (estatística) e `examples/generate_v2_audit_figures.py` (figuras da auditoria) → `paper_v2_campaign/`.

**Pendência conhecida:** a exportação de `delay_hist.csv` por slice (para CDF/CCDF por-pacote verdadeira) NÃO foi incluída no binário do piloto; as figuras usam âncoras de percentil por seed. Adicionar antes da campanha v2 completa se a CDF por-pacote for necessária ao paper.

**Bug do reassembly RLC UM — causa-raiz diagnosticada (2026-07-19):** o SIGABRT documentado em §14.3 tem mecanismo em 3 estágios, confirmado pelo stack trace dos evals failed da v1 (ex.: `congestion/seed=1/sa_eval_0052`): (1) sob perda pesada de segmentos, `NrRlcUm::ReassembleAndDeliver`/`ReassembleSnInterval` (`contrib/nr/model/nr-rlc-um.cc`) entrega um SDU corrompido em vez de descartá-lo; (2) o stack IP do UE vê destino-lixo não-local e, com IP forwarding habilitado (default ns-3), tenta ROTEAR o pacote de volta via NAS; (3) `NrQosRuleClassifier::Classify` (`nr-qos-rule-classifier.cc:142`) confia no payloadSize do cabeçalho corrompido e lê além do buffer → `NS_ASSERT` em `Buffer::PeekU8` → SIGABRT. **Mitigação aplicada no rslaq-sim.cc (não toca contrib/nr):** `IpForward=false` nos UEs (são hosts terminais; o SDU corrompido passa a ser descartado no RouteInput). A causa-raiz no módulo 5G-LENA permanece; a mitigação corta o caminho do crash. Runner segue tolerante (failed=hard-penalty) como defesa em profundidade.

### 14.6 Dashboard de monitoramento

Existe um dashboard Streamlit read-only para acompanhar a campanha em tempo real:

- **Arquivo**: `ns-o-ran-gym/examples/dashboard.py`.
- **Como rodar**: usar o venv dedicado em `ns-o-ran-gym/.venv-dashboard/` (PEP 668 bloqueia o python3 do sistema):
  ```
  cd ns-o-ran-gym
  .venv-dashboard/bin/streamlit run examples/dashboard.py
  ```
  Abre em `http://localhost:8501`, com auto-refresh a cada 15s.
- **O que mostra**: % global de conclusão e ETA (Seção A); heatmap cenário×seed e tabela de progresso por método (B); curvas de convergência, boxplots e ranking comparativo de GA/PSO/SA/híbrida por par selecionado (C); melhor candidato por cenário com pesos ótimos (D); ticker de atividade recente e detecção de erros nos logs (E).
- **Fonte de dados**: lê apenas sidecars `candidate.json`, `best_candidate_*.json`, `metaheuristic_results_*.csv` e o `batch_manifest.csv` da Phase 1. **Nunca modifica a campanha** (read-only).
- **Contagens embutidas** (iter=12, pop=6): GA=72, PSO=72, SA=72, híbrida=84 → 300 evals/par, 4500 avaliações globais para 15 pares cenário×seed. As constantes executáveis `EVALS_PER_METHOD` do `dashboard.py` já refletem esses valores; manter também comentários e textos do dashboard sincronizados se os hiperparâmetros mudarem.
- **Limitação conhecida**: o `streamlit-autorefresh` opcional não está instalado; o app usa fallback via `<meta http-equiv="refresh">` (recarrega a página no browser a cada 15s). Para auto-refresh nativo (sem reload visível), instale `streamlit-autorefresh` no venv-dashboard.

### 14.8 Campanha RSLAQ DDQN paper-faithful — Phase 4 (2026-08-11)

Baseline DRL do paper. Treina o agente DDQN paper-faithful (Yungaicela-Naula et al., IEEE TMC 2026) sobre o binário v2 (com fix do PDCP §14.3 e percentis por-pacote), para comparar diretamente com as meta-heurísticas e baselines da campanha v2 sob o mesmo protocolo (SIM_TIME=5, score Path C).

**Fidelidade ao paper (Hyp-set3, Table VI):** LR=0.001, γ=0.80, λϵ=0.998, ϵ_min=0.05, L=500, btsz=350, nsut=200, ntsr=100, E≈3500 interaction steps (= 35 episódios × 100 steps), psta=0.5, ω=[0.3333, 0.4000, 0.2667], reward piecewise Eq. 12 (não soma ponderada), arquitetura 4×Conv2D+BN+Tanh+FC, action space discreto 198 ações (66 pesos × 3 schedulers). O wrapper sobrescreve `DDQN_INTERACTION_STEPS=3500` e `DDQN_BUFFER_SIZE=500`/`DDQN_BATCH_SIZE=350` (defaults do `run_all_scenarios.sh` eram 20000/10000/64, divergentes do paper).

**Escopo vigente (decisão 2026-08-11):** 3 cenários leves (`low_traffic`, `normal`, `stressed`) × 5 seeds (1-5). `congestion` e `insufficient_resources` **adiados** — custo proibitivo nesta máquina (~13-15 dias/seed em congestion com treino online IPC; §14.7). Serão abordados posteriormente, possivelmente via offline-RL.

**Wrapper:** `ns-o-ran-gym/examples/run_rslaq_ddqn_paper_20260811.sh`. Invoca `run_all_scenarios.sh` com `RUN_TAG=20260811_rslaq_ddqn_paper`, `RUN_RSLAQ_DDQN_PAPER=1`, `DDQN_DRL_JOBS=5` (paralelismo). Output root: `results_controlled/heuristics_metaheuristics/20260811_rslaq_ddqn_paper/rslaq_ddqn_paper/`.

**Correção de path aplicada:** `run_controlled_rslaq_validation.sh:17` tinha `REPO_ROOT=/home/elioth/...` (usuário errado, sem `luy`); corrigido para `/home/eliothluy/...`. Sem essa correção, a Phase 4 falhava com `cd: arquivo inexistente`.

**Protocolo de métrica dupla (para comparação com meta-heurísticas):**
- **Primária (comparável):** score v2 Path C. Como o DDQN online produz `step_metrics.csv`/`ddqn_summary.json` na escala reward paper (não comparável), extrai-se a política greedy do checkpoint (`extract_ddqn_weights.py`), re-roda-se ns-3 standalone com `--baselineMode=slice_custom --weights=<w_embb,w_urllc,w_mtc>` (`rescore_ddqn_standalone.py`), e pontua-se com `score_summary_rows_v2`. Resultado: `ddqn_scored_v2.csv`, mesmo formato que `v2_best_candidates.csv`.
- **Secundária (fiel ao paper):** reward paper Eq. 8/12 (curva de treino em `ddqn_training_log.csv`), reportada no apêndice como convergência paper-faithful, **não** na tabela comparativa principal.

**Pipeline pós-treino:**
1. `extract_ddqn_weights.py` — carrega `ddqn_best.pt`, roda política greedy (ε=0), extrai pesos médios por (cenário, seed) → `ddqn_extracted_weights.csv`.
2. `rescore_ddqn_standalone.py` — re-roda ns-3 standalone com os pesos extraídos, pontua com Path C → `ddqn_scored_v2.csv`.
3. `analyze_ddqn_vs_meta.py` — tabela unificada {GA, PSO, SA, híbrida, RSLAQ DDQN} + Wilcoxon pareado (n=5) DDQN vs cada meta + boxplot → `comparison_ddqn_vs_meta_v2.csv`, `wilcoxon_ddqn_vs_meta.csv`, `fig_ddqn_vs_meta.pdf`.

**Lacunas e limitações:**
- O paper RSLAQ **não reporta número de seeds** (gap de reprodutibilidade); adotamos n=5 (mínimo para Wilcoxon válido).
- Treino online via IPC (semáforo POSIX + CSV) é ~15× mais caro que a máquina original do paper; low_traffic ~50 min/seed, normal ~2.5 h/seed, stressed ~8-12 h/seed.
- `btsz=350` com `L=500` é apertado (batch quase do tamanho do buffer) — fiel ao paper Table VI, mas pode degradar estabilidade do gradiente; documentar.
- Não há modelo pré-treinado paper-faithful no disco (os `.pt` em `models/` são offline-RL, linha MODA, não RSLAQ).

### 14.9 xApp DRL offline a partir das meta-heurísticas — método Bordin et al. (2026-08-18, sancionada pelo autor)

**Nova linha sancionada explicitamente pelo autor** (exceção ao §13, que proíbe deslocar o foco para DRL por iniciativa de agente): uma **xApp própria** com agente DRL **treinado offline a partir dos resultados das meta-heurísticas e heurísticas**, seguindo o método de *Bordin, Lacava, Polese, Cuomo, Melodia — "Design and Evaluation of Deep Reinforcement Learning for Energy Saving in Open RAN" (IEEE CCNC 2025, arXiv:2410.14021)* e do follow-up *"Enabling DRL Research for Energy Saving in Open RAN" (CCNC 2025 demo, arXiv:2601.02240; PDF local: `~/Documentos/openRAn/2601.02240v2.pdf` e `~/Documentos/openRAn/artigosBase/Design_and_Evaluation...pdf`)* — mas aplicado a **pesos de slice** em vez de energy saving. Esta linha absorve e evolui a linha MODA existente (§ CLAUDE.md "Offline DRL xApp"); o firewall de escopo permanece: o PRIMEIRO paper continua sendo o das meta-heurísticas.

**Princípio do método Bordin (o que copiamos):**
1. **Treinar offline, inferir online** — aderente à recomendação O-RAN WG2 AI/ML (não treinar online em rede viva); a exploração acontece no simulador, não na rede.
2. **Dataset de exploração gerado por HEURÍSTICAS seguras** — no paper, 4 políticas de sleep (Random, Static, Dynamic, Always On) × 3 configurações × 2 posicionamentos de UEs ≈ 3.000 simulações de 10 s → >300k data points. No nosso caso, as heurísticas de exploração são: as heurísticas adaptativas já implementadas (`slice_aqps`, `slice_demand_greedy`, `slice_sla_greedy`, `slice_least_waste`, `slice_qos_mixed`, `slice_random_vine`, `slice_meta_risk_elastic`) + **as trajetórias de busca das próprias meta-heurísticas** (todas as avaliações GA/PSO/SA/híbrida já logadas com pesos+KPIs).
3. **Offline RL conservativo** — o DQN do paper usa **REM-DQN + CQL** (Random Ensemble Mixture + Conservative Q-Learning) para não extrapolar o suporte do dataset; alternativa PPO offline (Stable-Baselines3, batch 256, entropy 0.001-0.003).
4. **Normalização cuidadosa** — quantile transformer (Normal/Uniform) nos componentes do reward; KPMs do estado em [0,1]. Destacado pelos autores como essencial (dados power-law).
5. **Seleção de features por correlação com o reward** (12 KPMs/célula + 1 global no paper).
6. **Avaliação em malha fechada** no simulador, seeds estatisticamente independentes do treino, CDFs por KPM + fronteira de trade-off; baselines = as próprias heurísticas que geraram o dataset.
7. **xApp no Near-RT RIC**, decisão na periodicidade de indicação (100 ms lá; nossa periodicidade natural = 10 ms de frame, mas o `slice_custom` offline é estático — ver adaptações abaixo).

**Adaptações necessárias ao nosso problema (diferenças declaradas):**
- **Ação**: pesos de slice contínuos no simplexo (3-D) ou discretizados (66 ações step 0.1, como RSLAQ) — não bits on/off. AActionController existente (`nsoran/action_controller.py`) já converte pesos em `dedicatedPRB` por slice.
- **Reward**: usar o **score Path C** como recompensa (mesma escala das meta-heurísticas) — ou versão contínua dele (termo de viabilidade + fairness), jamais o reward paper RSLAQ (escalas não comparáveis; §14.8). Peso de trade-off estilo Tabela I do Bordin: w·SLA vs w·eficiência.
- **Estado**: KPMs por slice (throughput, delay p99, PDR, buffers, satisfação composta) + carga/cenário; normalizar [0,1].

**Gap crítico da linha atual (MODA) → o que evoluir:** o dataset atual (`offline_dataset_v2.parquet`, 3.722 rows) **não tem next-state** (avaliações standalone independentes) — por isso os modelos ddqn/sac/ppo atuais são honestamente rotulados de **contextual-bandit/supervised** (Q-regression/BC/RWR), NÃO offline RL com TD. Para implementar o método Bordin de fato:
1. **Construir transições (s, a, r, s')**: as fontes viáveis são (a) os `timeseries.csv`/`slice_alloc.csv` das runs existentes (KPIs por período de 10 ms dentro de cada simulação — janelas consecutivas formam transições), e/ou (b) novas coletas com heurísticas dinâmicas rodando via IPC (como o treino RSLAQ, ~3500 steps/run), logando a sequência de decisões.
2. **Treinar offline RL pleno**: REM-DQN+CQL e/ou PPO offline sobre as transições; manter os bandits atuais como baselines internos da linha.
3. **Avaliação closed-loop**: re-executar a política inferida via `eval_offline_rl_closedloop.py` (protocolo já validado, Path C) contra {meta-heurísticas best, RSLAQ DDQN, heurísticas}, seeds disjuntas do treino.

**Regras de honestidade (herdadas e reforçadas):**
- Nomear os modelos pelo que são: `rem_cql_dqn`, `ppo_offline`, e os bandits legados `q_regression`/`bc_top20`/`rwr` — nunca chamar bandit de "DRL".
- Nunca misturar escalas de reward sem rótulo; a métrica de comparação primária é Path C.
- Runs `pure_rr/pf/bcqi` continuam EXCLUÍDAS do dataset (sem semântica de pesos; envenenaria a regressão).
- Baselines fechados: as heurísticas que geram o dataset DEVEM também aparecer na avaliação closed-loop (auto-consistência do método Bordin).
- Diversidade do dataset é requisito (>= 3 políticas de exploração × múltiplos cenários/seeds) antes de qualquer claim de generalização.

**Artefatos existentes reaproveitáveis:** `build_offline_dataset.py`/`offline_dataset.py`, `train_offline_rl.py`/`offline_models.py`, `eval_offline_rl_closedloop.py`, `xapp_slice_optimizer.py`, `models/closedloop_eval/` (protocolo de avaliação com 72/72 células n=3), teste `test_scoring_v2.py`. O `es_env.py`/`datalake.py` do ns-o-ran-gym carregam o esqueleto Bordin original (KPMs cell-centric fora do Datalake UE-cêntrico — mesma limitação que teremos com KPMs slice-cêntricos).
