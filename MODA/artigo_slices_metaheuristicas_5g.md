# Otimização Contínua de Pesos de Alocação para Network Slicing em Redes 5G/O-RAN usando Meta-Heurísticas

## Resumo

Este artigo investiga o uso de meta-heurísticas para otimizar a distribuição de recursos entre slices em redes 5G/O-RAN. O problema é formulado como uma otimização contínua, em que uma solução candidata é um vetor de pesos de alocação para os slices eMBB, URLLC e MTC. A avaliação experimental utiliza simulações em ns-3/ns-O-RAN Gym e compara Particle Swarm Optimization (PSO), Genetic Algorithm (GA), Simulated Annealing (SA) e uma meta-heurística híbrida que combina exploração populacional, operadores genéticos, atualização inspirada em PSO e refinamento local estocástico. Os resultados da campanha `20260623_123740` mostram que o PSO obteve a melhor solução nos cenários normal e congestionado, com scores de 68,085 e 84,265, respectivamente. A meta-heurística híbrida obteve a melhor solução no cenário de baixo tráfego, com score de 56,818. Considerando os três cenários comparáveis, o PSO foi o método mais consistente em termos de melhor solução encontrada, enquanto a híbrida apresentou desempenho competitivo, mas não superior de forma geral. A comparação pareada com heurísticas de alocação na semente 1 indica ganhos pequenos em baixo tráfego, moderados no cenário normal e relevantes em congestionamento. Entretanto, a evidência estatística ainda é limitada, pois a busca meta-heurística disponível não contém múltiplas execuções independentes por método e os baselines puros RR, BCQI, PF, AQPS e RSLAQ não estão completos no mesmo protocolo experimental da campanha analisada.

**Palavras-chave:** network slicing; 5G; O-RAN; meta-heurísticas; PSO; algoritmo genético; simulated annealing; alocação de recursos.

## 1. Introdução

Redes 5G introduzem suporte nativo a network slicing, permitindo que diferentes classes de serviço compartilhem a mesma infraestrutura física sob requisitos heterogêneos de vazão, latência, confiabilidade e escala de conectividade. Em um cenário típico, slices eMBB priorizam alta vazão, slices URLLC priorizam latência e confiabilidade, e slices MTC priorizam conectividade massiva e atendimento de tráfego distribuído. Essa heterogeneidade torna a alocação de recursos de rádio um problema de decisão multiobjetivo e dependente do estado da rede.

Arquiteturas O-RAN favorecem o uso de controladores programáveis, xApps e loops de controle para ajustar decisões de rádio de forma automatizada. Nesse contexto, uma estratégia prática consiste em manter o scheduler intra-slice tradicional e otimizar, no nível inter-slice, pesos que controlam a partição do orçamento de recursos entre fatias. Essa separação preserva a compatibilidade com schedulers conhecidos e reduz a complexidade do problema de decisão: em vez de redefinir o scheduler, a busca procura um vetor contínuo de pesos capaz de produzir bom compromisso entre satisfação de SLA, atendimento de carga, entrega de pacotes e utilização de recursos.

Este trabalho avalia meta-heurísticas como mecanismos de busca para esse vetor de pesos. O foco não é propor um novo scheduler independente, mas analisar como diferentes métodos de otimização exploram o espaço contínuo de alocação de slices. As meta-heurísticas avaliadas são PSO, GA, SA e uma abordagem híbrida reformulada. AQPS e RSLAQ são mantidos como bases conceituais e comparativas do estudo: AQPS motiva alocação sensível a QoS em cenários multi-serviço, enquanto RSLAQ fornece a perspectiva de controle orientado a SLA em O-RAN. Os schedulers puros RR, BCQI e PF devem permanecer como baselines de comparação na versão experimental final.

As contribuições deste artigo são:

1. Formular a distribuição inter-slice como um problema de otimização contínua sobre pesos de alocação.
2. Avaliar PSO, GA, SA e uma meta-heurística híbrida usando uma função objetivo comum baseada em KPIs de rede e SLA.
3. Comparar as melhores soluções encontradas em cenários de baixo tráfego, carga normal e congestionamento.
4. Identificar limitações estatísticas e experimentais que devem ser corrigidas antes de uma submissão final.

## 2. Formulação do Problema

Considere um conjunto de \(S\) slices. Neste estudo, \(S=3\), correspondendo aos slices eMBB, URLLC e MTC. Uma solução candidata é representada pelo vetor contínuo:

\[
\mathbf{w} = [w_{eMBB}, w_{URLLC}, w_{MTC}].
\]

Cada componente \(w_i\) define o peso de alocação associado ao slice \(i\). A implementação normaliza os pesos para respeitar:

\[
w_i \geq 0, \quad \forall i \in \{1,\ldots,S\},
\]

\[
\sum_{i=1}^{S} w_i = 1.
\]

Assim, as meta-heurísticas operam em um domínio contínuo simples, mas sujeito a normalização. Quando um vetor candidato contém valores negativos ou soma inadequada, a função `normalize_weights` da implementação remove valores negativos, completa dimensões ausentes e renormaliza o vetor. Caso a soma seja nula, utiliza-se a alocação uniforme \([1/3, 1/3, 1/3]\).

A qualidade de cada vetor é avaliada por simulação. O problema é tratado como maximização:

\[
\max_{\mathbf{w}} F(\mathbf{w}),
\]

em que \(F(\mathbf{w})\) é o score calculado a partir dos KPIs agregados por slice. A função objetivo implementada combina satisfação média de SLA, pior satisfação de SLA entre slices, atendimento médio da carga oferecida, PDR médio e utilização média do orçamento. Penalidades adicionais são aplicadas para atraso médio URLLC acima de 10 ms, starvation do slice MTC e ocupação de buffer no eMBB. Em forma resumida:

\[
F(\mathbf{w}) = 100(0{,}35\bar{s}_{SLA} + 0{,}20s_{SLA}^{min} + 0{,}20\bar{s}_{load} + 0{,}15\bar{p}_{PDR} + 0{,}10\bar{u}) - P_{URLLC} - P_{MTC} - P_{eMBB}.
\]

Essa função favorece soluções que atendem os três slices, evitando que alta vazão em um slice compense completamente a degradação dos demais. A definição de melhor solução neste artigo segue diretamente o maior valor observado de \(F(\mathbf{w})\).

## 3. Métodos Avaliados

### 3.1 PSO

O PSO mantém uma população de partículas, cada uma representando um vetor de pesos. A posição de cada partícula é atualizada por termos de inércia, melhor posição individual e melhor posição global. Após cada atualização, o vetor é normalizado. Na implementação analisada, o PSO utiliza coeficiente de inércia 0,55 e fatores cognitivo/social 1,30. A busca foi executada com quatro iterações e seis partículas, totalizando 24 avaliações por cenário.

### 3.2 GA

O GA usa uma população de vetores de pesos, seleção por elite, recombinação por crossover e mutação. Em cada geração, os indivíduos são avaliados pelo score de simulação; os melhores compõem a elite e geram a próxima população. Na campanha analisada, o GA também realizou 24 avaliações por cenário, decorrentes de quatro iterações e população de seis indivíduos.

### 3.3 SA

O SA inicia a busca a partir de um vetor uniforme e gera candidatos por mutação com temperatura decrescente. Candidatos piores podem ser aceitos com probabilidade dependente da diferença de score e da temperatura. Nesta execução, o SA realizou quatro avaliações por cenário, uma por iteração. Portanto, seus resultados devem ser interpretados com cautela, pois o orçamento de avaliação foi menor que o de GA, PSO e híbrida.

### 3.4 Meta-Heurística Híbrida

A abordagem híbrida combina elementos populacionais e locais. A população inicial inclui candidatos semeados, incluindo pesos uniformes e configurações estruturadas. A cada iteração, o método avalia partículas, mantém um arquivo de elite, aplica refinamento local por mutação do melhor global e aceita refinamentos por critério inspirado em SA. A próxima população é formada por elites, atualizações inspiradas em PSO, recombinações inspiradas em GA e mutações adaptativas. Na campanha analisada, a híbrida realizou 28 avaliações por cenário, pois cada uma das quatro iterações avaliou a população e um candidato local adicional.

## 4. Metodologia Experimental

Os resultados analisados estão em:

`/home/elioth/Documentos/artigo_jussi/ns-o-ran-gym/results_controlled/heuristics_metaheuristics/20260623_123740/`

A campanha contém resultados de meta-heurísticas para três cenários: `low_traffic`, `normal` e `congestion`. O bloco de heurísticas contém também `stressed` e `insufficient_resources`, mas esses cenários não possuem resultados meta-heurísticos agregados no mesmo diretório; por isso, não são usados na comparação principal entre meta-heurísticas.

As simulações utilizam um gNB, duplexação TDD, RLC UM, padrão TDD solicitado `D|D|8D|4GB|4U|U|U`, três slices e scheduler intra-slice PF para os modos slice-aware e `slice_custom`. A duração de simulação registrada nos metadados é de 10 s, com 0,2 s de drenagem.

Os cenários comparáveis são:

| Cenário | UEs | UEs por slice | Carga eMBB | Carga URLLC | Carga MTC |
|---|---:|---|---:|---:|---:|
| `low_traffic` | 10 | eMBB=2, URLLC=2, MTC=6 | 55 Mbps | 1 Mbps | 1 Mbps |
| `normal` | 20 | eMBB=5, URLLC=5, MTC=10 | 70 Mbps | 1 Mbps | 2 Mbps |
| `congestion` | 60 | eMBB=15, URLLC=10, MTC=35 | 180 Mbps | 5 Mbps | 60 Mbps |

O SLA mínimo configurado é 50 Mbps para eMBB, 1 Mbps para URLLC e 1 Mbps para MTC. A comparação principal usa o score da função objetivo. Métricas de rede por slice, como throughput, atraso, PDR, satisfação de carga e satisfação de SLA, são usadas para interpretar os scores.

## 5. Resultados

### 5.1 Comparação entre Meta-Heurísticas

A Tabela 1 resume os scores por método e cenário. Os valores de média, mediana e desvio padrão são calculados sobre avaliações de função objetivo dentro de uma única busca, não sobre execuções independentes. Assim, o melhor score é a métrica primária de qualidade de solução; as estatísticas de dispersão indicam apenas o comportamento interno da busca.

**Tabela 1. Estatísticas dos scores das meta-heurísticas.**

| Cenário | Método | Avaliações | Melhor | Média | Mediana | Desvio | Pior | Pesos do melhor candidato (eMBB, URLLC, MTC) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `low_traffic` | GA | 24 | 56,676 | 48,285 | 56,506 | 19,704 | -23,688 | (0,669; 0,135; 0,195) |
| `low_traffic` | PSO | 24 | 56,568 | 52,635 | 56,355 | 9,987 | 20,284 | (0,571; 0,169; 0,259) |
| `low_traffic` | SA | 4 | 56,043 | 55,757 | 55,846 | 0,336 | 55,292 | (0,308; 0,197; 0,495) |
| `low_traffic` | Híbrida | 28 | **56,818** | 53,820 | 56,504 | 14,122 | -18,218 | (0,692; 0,174; 0,135) |
| `normal` | GA | 24 | 67,479 | 56,159 | 61,830 | 11,812 | 37,708 | (0,572; 0,092; 0,336) |
| `normal` | PSO | 24 | **68,085** | 57,376 | 63,021 | 19,990 | -25,780 | (0,753; 0,019; 0,228) |
| `normal` | SA | 4 | 62,108 | 61,182 | 61,296 | 0,885 | 60,029 | (0,278; 0,119; 0,603) |
| `normal` | Híbrida | 28 | 67,246 | 62,644 | 66,152 | 8,780 | 38,167 | (0,607; 0,127; 0,266) |
| `congestion` | GA | 24 | 82,991 | 71,898 | 78,156 | 13,051 | 41,468 | (0,526; 0,118; 0,355) |
| `congestion` | PSO | 24 | **84,265** | 65,398 | 74,852 | 28,194 | -44,472 | (0,490; 0,062; 0,448) |
| `congestion` | SA | 4 | 81,869 | 70,362 | 69,427 | 8,718 | 60,723 | (0,399; 0,214; 0,387) |
| `congestion` | Híbrida | 28 | 81,953 | 75,380 | 77,739 | 9,319 | 43,209 | (0,424; 0,203; 0,373) |

No cenário `low_traffic`, a híbrida encontrou o melhor score, mas a diferença para GA e PSO foi pequena: 0,142 ponto acima do GA e 0,250 ponto acima do PSO. Esse cenário mostra uma região de soluções de desempenho muito próximo, na qual pequenas variações de peso alteram pouco o score final.

No cenário `normal`, o PSO apresentou o melhor score, 68,085, seguido por GA, 67,479, e híbrida, 67,246. A diferença entre os três melhores métodos foi inferior a 0,9 ponto, mas o SA ficou abaixo, com melhor score de 62,108. A melhor solução do PSO deslocou a maior parte do peso para eMBB, mas manteve parcela relevante para MTC, sacrificando URLLC sem violar totalmente sua satisfação de SLA.

No cenário `congestion`, o PSO apresentou a melhor solução global da campanha, com score de 84,265. O GA alcançou 82,991 e a híbrida 81,953. A superioridade do PSO foi mais clara nesse cenário, em que o vetor encontrado alocou aproximadamente 49,0% para eMBB, 6,2% para URLLC e 44,8% para MTC. Essa configuração reflete a necessidade de sustentar MTC sob carga elevada, sem abandonar completamente eMBB e URLLC.

### 5.2 Métricas dos Melhores Candidatos

A Tabela 2 apresenta os KPIs por slice para os melhores candidatos de cada cenário.

**Tabela 2. KPIs por slice dos melhores candidatos meta-heurísticos.**

| Cenário | Método | Slice | Throughput | Atraso médio | PDR | Atendimento da carga | Satisfação de SLA | Utilização |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `low_traffic` | Híbrida | eMBB | 28,012 Mbps | 20,439 ms | 50,00% | 50,93% | 56,02% | 100,00% |
| `low_traffic` | Híbrida | URLLC | 1,560 Mbps | 4,969 ms | 100,00% | 100,00% | 100,00% | 100,00% |
| `low_traffic` | Híbrida | MTC | 0,682 Mbps | 8,210 ms | 53,33% | 68,25% | 68,25% | 100,00% |
| `normal` | PSO | eMBB | 24,388 Mbps | 1112,893 ms | 34,20% | 34,84% | 48,78% | 100,00% |
| `normal` | PSO | URLLC | 0,938 Mbps | 14,864 ms | 60,17% | 93,84% | 93,84% | 100,00% |
| `normal` | PSO | MTC | 1,843 Mbps | 3,865 ms | 72,04% | 92,18% | 100,00% | 100,00% |
| `congestion` | PSO | eMBB | 81,115 Mbps | 1754,333 ms | 44,24% | 45,06% | 100,00% | 100,00% |
| `congestion` | PSO | URLLC | 4,479 Mbps | 5,981 ms | 70,00% | 89,59% | 100,00% | 100,00% |
| `congestion` | PSO | MTC | 39,878 Mbps | 19,592 ms | 62,95% | 66,46% | 100,00% | 100,00% |

Os resultados evidenciam que o score não é equivalente a throughput total. No cenário `normal`, por exemplo, o PSO aceita baixa satisfação do eMBB para preservar URLLC e MTC. Em `congestion`, o melhor vetor aumenta a participação do MTC, reduzindo risco de starvation e elevando o score mesmo com PDR ainda limitado. Portanto, a função objetivo favorece compromisso entre slices, não maximização isolada de vazão.

### 5.3 Comparação com Heurísticas de Alocação no Mesmo Diretório

Além das meta-heurísticas, a campanha solicitada contém heurísticas de alocação inter-slice: `slice_demand_greedy`, `slice_sla_greedy`, `slice_least_waste`, `slice_qos_mixed` e `slice_random_vine`. Como a busca meta-heurística agregada disponível foi executada apenas com `seed=1`, a comparação mais direta usa a mesma semente. A Tabela 3 compara o melhor candidato meta-heurístico com a melhor heurística da semente 1.

**Tabela 3. Melhor meta-heurística versus melhor heurística na semente 1.**

| Cenário | Melhor meta-heurística | Score meta | Melhor heurística seed=1 | Score heurística | Ganho absoluto | Ganho relativo |
|---|---|---:|---|---:|---:|---:|
| `low_traffic` | Híbrida | 56,818 | `slice_demand_greedy` | 56,654 | +0,164 | +0,3% |
| `normal` | PSO | 68,085 | `slice_demand_greedy` | 65,813 | +2,272 | +3,5% |
| `congestion` | PSO | 84,265 | `slice_random_vine` | 70,988 | +13,277 | +18,7% |

Os ganhos são quase nulos em `low_traffic`, moderados em `normal` e expressivos em `congestion`. Isso sugere que a otimização contínua dos pesos é mais útil quando a rede está sob disputa intensa por recursos. Em baixo tráfego, o espaço de soluções contém vários vetores com score semelhante, reduzindo a vantagem prática de uma busca mais sofisticada.

A Tabela 4 mostra, em complemento, as estatísticas das heurísticas sobre três sementes. Esses valores não devem ser comparados diretamente com as meta-heurísticas como teste estatístico, pois as meta-heurísticas ainda não foram repetidas nas mesmas três sementes.

**Tabela 4. Scores das heurísticas no diretório analisado, agregados sobre três sementes.**

| Cenário | Heurística | Melhor | Média | Desvio |
|---|---|---:|---:|---:|
| `low_traffic` | `slice_demand_greedy` | 99,163 | 71,821 | 23,726 |
| `low_traffic` | `slice_sla_greedy` | 99,160 | 71,497 | 23,996 |
| `low_traffic` | `slice_qos_mixed` | 99,161 | 71,378 | 24,086 |
| `low_traffic` | `slice_random_vine` | 99,001 | 71,503 | 23,893 |
| `low_traffic` | `slice_least_waste` | 98,932 | 70,840 | 24,377 |
| `normal` | `slice_sla_greedy` | 97,997 | 78,656 | 17,049 |
| `normal` | `slice_qos_mixed` | 96,115 | 77,761 | 16,295 |
| `normal` | `slice_random_vine` | 97,981 | 76,818 | 19,290 |
| `normal` | `slice_demand_greedy` | 89,238 | 75,753 | 12,108 |
| `normal` | `slice_least_waste` | 82,108 | 72,988 | 8,758 |
| `congestion` | `slice_random_vine` | 70,988 | 68,533 | 3,697 |
| `congestion` | `slice_sla_greedy` | 54,486 | 53,881 | 0,552 |
| `congestion` | `slice_qos_mixed` | 54,268 | 52,976 | 1,388 |
| `congestion` | `slice_demand_greedy` | 52,112 | 51,623 | 0,565 |
| `congestion` | `slice_least_waste` | 50,854 | 49,977 | 0,828 |

A alta variabilidade das heurísticas em `low_traffic` e `normal` mostra que a semente afeta fortemente os resultados. Portanto, a conclusão sobre superioridade das meta-heurísticas deve permanecer restrita: elas superam as heurísticas na comparação pareada com `seed=1`, mas ainda não há base para afirmar superioridade estatística geral.

## 6. Discussão

Os resultados apontam três conclusões técnicas.

Primeiro, o PSO encontrou as melhores soluções em dois dos três cenários comparáveis. A melhor solução global foi obtida em congestionamento, com score 84,265. Isso sugere que a dinâmica de melhor pessoal e melhor global do PSO foi eficaz na exploração do espaço contínuo de pesos quando o trade-off entre eMBB, URLLC e MTC se tornou mais severo.

Segundo, a meta-heurística híbrida foi competitiva, mas não dominante. Ela venceu no cenário `low_traffic`, mas a margem foi muito pequena. Em `normal` e `congestion`, ficou abaixo do PSO. Sua média de score foi alta em alguns cenários, mas essa média reflete a trajetória de uma busca com 28 avaliações, não execuções independentes. Assim, não se deve concluir que a híbrida é estatisticamente superior sem novas rodadas.

Terceiro, o ganho da otimização por pesos depende da condição de carga. Em congestionamento, o melhor PSO superou a melhor heurística da semente 1 em 18,7%. Esse é o resultado mais relevante da campanha, pois indica que a busca contínua consegue encontrar uma partição de recursos mais adequada do que regras manuais quando há forte competição entre slices. Em baixo tráfego, o ganho foi de apenas 0,3%, o que sugere que heurísticas simples podem ser suficientes quando a pressão por recursos é menor.

Um ponto metodológico importante é que os resultados ainda não fecham a comparação completa exigida para o artigo final. O diretório analisado não contém, no mesmo protocolo de 10 s e com os mesmos artefatos meta-heurísticos, os schedulers puros RR, BCQI e PF, nem comparações completas com AQPS e RSLAQ. Há execuções posteriores no workspace com esses baselines, mas elas usam outro protocolo experimental e não devem ser misturadas sem controle. Para uma versão de submissão, RR, BCQI, PF, AQPS e RSLAQ devem ser reexecutados ou consolidados no mesmo conjunto de cenários, duração, sementes e função objetivo.

## 7. Ameaças à Validade

As principais ameaças à validade são:

1. A busca meta-heurística analisada foi registrada para `seed=1`; não há 30 execuções independentes por método.
2. O orçamento de avaliação difere entre métodos: GA e PSO têm 24 avaliações, SA tem 4 e a híbrida tem 28.
3. As médias das meta-heurísticas são médias de avaliações dentro de uma busca, não médias de execuções independentes.
4. Os cenários `stressed` e `insufficient_resources` não possuem resultados meta-heurísticos agregados no diretório solicitado.
5. RR, BCQI, PF, AQPS e RSLAQ não estão completos no mesmo protocolo da campanha `20260623_123740`.
6. Não foram aplicados testes de Friedman, Wilcoxon ou Nemenyi, pois os dados disponíveis não constituem amostras pareadas independentes suficientes.

Essas limitações não invalidam a análise exploratória, mas impedem afirmações fortes de significância estatística.

## 8. Conclusão

Este artigo analisou a otimização contínua dos pesos de alocação de slices em redes 5G/O-RAN usando PSO, GA, SA e uma meta-heurística híbrida. Nos resultados disponíveis da campanha `20260623_123740`, o PSO foi a meta-heurística que encontrou as melhores soluções de forma mais consistente: venceu nos cenários `normal` e `congestion` e obteve o maior score global, 84,265, no cenário congestionado. A meta-heurística híbrida venceu em `low_traffic`, com score 56,818, mas por margem pequena em relação ao GA e ao PSO.

Na comparação pareada com heurísticas de alocação usando `seed=1`, a melhor meta-heurística superou a melhor heurística em todos os três cenários comparáveis. O ganho foi marginal em baixo tráfego, moderado em carga normal e expressivo em congestionamento. Portanto, a principal evidência experimental é que a otimização meta-heurística dos pesos é mais valiosa sob alta competição por recursos.

A conclusão científica deve ser formulada com cautela: PSO é o melhor método segundo o critério de melhor solução encontrada nos dados analisados, enquanto a híbrida é competitiva, mas ainda não comprovadamente superior. Para a versão final do artigo, é necessário executar múltiplas rodadas independentes por método, equalizar o orçamento de avaliações e incluir RR, BCQI, PF, AQPS e RSLAQ no mesmo protocolo experimental.

## Referências a Consolidar

1. RSLAQ: Robust SLA-driven 6G O-RAN QoS xApp Using Deep Reinforcement Learning.
2. AQPS: Adaptive QoS-Aware Priority Scheduling Solution for Dynamic Radio Resource Allocation in Multiservice 5G Network Slicing Environments.
3. Literatura base sobre O-RAN, xApps, 5G network slicing, PSO, GA e Simulated Annealing.

As referências acima devem ser convertidas para o formato bibliográfico exigido pelo veículo de submissão após confirmação dos metadados completos dos PDFs e documentos do workspace.
