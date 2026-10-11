# Fatiamento Seguro de RAN em O-RAN: Minimização de Violações de SLA via Aprendizado por Reforço Baseado em Modelos

*Aerman Tuerxun — Graduate School of Engineering, The University of Tokyo, Tokyo, Japão*
*Akihiro Nakao — Graduate School of Engineering, The University of Tokyo, Tokyo, Japão*

## Resumo

O fatiamento de RAN em arquiteturas Open RAN (O-RAN) exige uma alocação de recursos dinâmica e inteligente para garantir tanto uma alta eficiência no uso de recursos quanto o cumprimento rigoroso dos Acordos de Nível de Serviço (SLAs). No entanto, métodos convencionais de aprendizado por reforço (RL) enfrentam dificuldades nesse cenário devido ao seu comportamento de exploração inseguro, resultando frequentemente em violações de SLA. Neste artigo, propomos uma estrutura de aprendizado por reforço baseado em modelo (MBRL) segura e eficiente em termos amostrais, projetada especificamente para o fatiamento de RAN sensível a SLAs. Nosso agente utiliza um regressor quantílico kernelizado online para estimar o quantil inferior de indicadores-chave de desempenho (KPIs), permitindo a modelagem direta do risco de violação de SLA. Para garantir a segurança durante a adaptação online, a política incorpora uma margem de segurança ciente da incerteza, promovendo decisões conservadoras sob alta incerteza do modelo. Simulações extensivas demonstram que nosso método reduz as violações de SLA para menos de 1%, mantendo, ao mesmo tempo, uma alta eficiência de recursos e alcançando uma convergência rápida e estável.

**Termos de Indexação — O-RAN, Fatiamento de RAN, Alocação Segura de Recursos, Aprendizado Online, Regressão Quantílica, Método de Kernel.**

## I. Introdução

O advento do 5G e a evolução para redes 6G prometem dar suporte a um ecossistema diversificado de serviços que vai muito além da simples banda larga móvel [1]. Esses serviços — incluindo Banda Larga Móvel Aprimorada (eMBB) para streaming de alta definição, Comunicações Ultraconfiáveis e de Baixa Latência (URLLC) para sistemas autônomos e a internet tátil, e Comunicações Massivas do Tipo Máquina (mMTC) para a Internet das Coisas — apresentam requisitos de desempenho conflitantes. Uma arquitetura de rede única e monolítica não consegue atender eficientemente a essas necessidades díspares de forma simultânea. O fatiamento de rede (*network slicing*) surgiu como a principal solução arquitetônica, permitindo que operadoras de rede criem múltiplas redes virtuais de ponta a ponta, logicamente isoladas, sobre uma infraestrutura física comum. Cada fatia pode ser configurada e otimizada independentemente para as demandas específicas do serviço que ela suporta [2].

Para concretizar todo o potencial do fatiamento de rede, a rede subjacente deve ser aberta, programável e inteligente. O paradigma da Rede de Acesso por Rádio Aberta (O-RAN) atende a essa necessidade ao desagregar a RAN e introduzir um novo nó lógico: o Controlador Inteligente da RAN (RIC) [3]. O RIC foi projetado para hospedar aplicações de terceiros (conhecidas como xApps e rApps) capazes de utilizar IA e aprendizado de máquina para gerenciar e otimizar recursos de rede em tempo quase real, possibilitando níveis sem precedentes de automação e eficiência [4]. Essa transição da configuração estática para um controle inteligente e orientado por dados é fundamental para proporcionar a flexibilidade prometida pelo fatiamento de rede.

Essa nova capacidade arquitetônica traz consigo um desafio crítico: uma das funções mais vitais para uma xApp no RIC é a alocação dinâmica de recursos de rádio, como Blocos de Recursos Físicos (PRBs) [5], entre as diversas fatias de rede (*network slices*). Essa tarefa envolve um *trade-off* complexo. Por um lado, o controlador deve garantir que cada fatia receba recursos suficientes para atender ao seu Acordo de Nível de Serviço (SLA) específico e rigoroso. Por outro lado, para assegurar a viabilidade comercial, ele deve maximizar a eficiência espectral global, evitando o provisionamento excessivo de recursos. Surge, então, o problema central: como projetar um agente inteligente capaz de aprender, de forma online, uma política de alocação de recursos ótima que satisfaça ambos os objetivos conflitantes.

O Aprendizado por Reforço (RL) emergiu como uma ferramenta poderosa para a tomada de decisões dinâmicas em ambientes de rede complexos. No entanto, os métodos tradicionais de RL livre de modelo (*model-free* ou MFRL) — como Q-learning e PPO — dependem intrinsecamente de uma extensa exploração por tentativa e erro, o que não apenas leva a atualizações de política inseguras, mas também resulta em violações frequentes e graves dos SLAs. Além disso, a elevada complexidade amostral e o custo computacional do MFRL dificultam sua aplicabilidade em plataformas RIC com recursos limitados. Em resposta, trabalhos recentes têm recorrido ao MBRL (*model-based* RL) para obter maior eficiência amostral e uma adaptação online mais segura. Contudo, grande parte da literatura existente sobre MBRL ainda se baseia em modelos excessivamente simplistas ou em premissas *offline*, muitas vezes carecendo de mecanismos para quantificar a incerteza ou minimizar explicitamente os riscos relacionados ao SLA.

Para esse fim, propomos um novo agente de aprendizado por reforço baseado em modelo (MBRL) com regressão quantílica online, projetado especificamente para uma operação segura e eficiente em termos amostrais no âmbito do RIC. Nossa abordagem aprende um modelo probabilístico do desempenho do sistema para estimar diretamente o risco de violação de SLA para qualquer alocação de recursos. A política de controle do agente utiliza, então, esse modelo aprendido para tomar decisões conscientes dos riscos, agindo com cautela diante da incerteza para garantir uma exploração segura e um desempenho estável. Isso permite que o sistema vise diretamente uma taxa de violação de SLA desejada, ao mesmo tempo em que otimiza a eficiência de recursos.

As principais contribuições deste trabalho são as seguintes:

- Propomos uma nova arquitetura de agente MBRL adequada para implantação em um RIC O-RAN, projetada para aprendizado online seguro em fatiamento de RAN.
- Introduzimos o uso de regressão quantílica online como o mecanismo central de aprendizado, permitindo que o agente modele e gerencie diretamente o risco de violações de SLA.
- Projetamos uma política de controle consciente de riscos que utiliza a incerteza do modelo para aplicar uma margem de segurança dinâmica, criando um sistema que equilibra a exploração segura com a eficiência de recursos.

O restante do artigo está organizado da seguinte forma. A Seção II discute os trabalhos relacionados. As Seções III e IV formulam o problema e a solução proposta. A Seção V avalia a proposta por meio de simulação do sistema e a compara com trabalhos relacionados. A Seção VI apresenta uma breve conclusão do artigo.

## II. Trabalhos Relacionados

A natureza dinâmica do gerenciamento de recursos em 5G e além tem impulsionado um interesse crescente na aplicação de RL. No contexto de fatiamento de RAN, a maioria dos trabalhos existentes adota abordagens MFRL, devido ao seu forte desempenho em tarefas de controle de alta dimensionalidade.

Uma tendência dominante é o uso de Redes Q Profundas (DQN) e suas variantes para o gerenciamento de recursos das fatias. Por exemplo, [6] emprega um agente *Double Dueling Q-learning* para controlar o compartilhamento e o escalonamento de recursos entre serviços, enquanto [7] introduz uma Rede Q Distribucional baseada em GAN (GAN-DDQN) para melhorar a eficiência espectral e a QoE. Esses métodos normalmente impõem as restrições de SLA por meio de modelagem de recompensa (*reward shaping*), tratando as violações como penalidades. No entanto, tal imposição “suave” (*soft*) não impede que o agente tome decisões de alto risco que podem resultar em violações graves de SLA — um resultado inaceitável para serviços de missão crítica.

Outros trabalhos exploram melhorias no projeto do agente ou na formulação do problema. [8] propõe um agente hierárquico de dupla escala de tempo para desacoplar decisões de curto e longo prazo, enquanto [9] utiliza um MDP competitivo do tipo *Anytime* para garantir SLAs de forma perpétua. Métodos como MicroOpt [10] e DRL guiado por explicação [11] afastam-se do RL de ponta a ponta (*end-to-end*), empregando otimização baseada em gradiente ou *pipelines* de XAI para maior transparência. Embora inovadoras, essas abordagens frequentemente introduzem complexidade arquitetural significativa e ainda dependem de mecanismos de recompensa-penalidade que carecem de garantias rígidas de segurança.

Nosso trabalho segue uma direção diferente ao propor um agente leve, online e baseado em modelo, que é ao mesmo tempo eficiente em termos amostrais e inerentemente seguro. O método anterior mais comparável é o KBRL, de Alcaraz et al. [12], que utiliza um método de kernel baseado em modelo para aumentar a eficiência amostral. No entanto, a dependência do KBRL em uma suposição de monotonicidade limita sua aplicabilidade em ambientes RAN reais, nos quais efeitos da camada física frequentemente introduzem comportamentos não monotônicos. Nosso agente remove essa suposição ao aprender um modelo de resposta de KPI mais refinado e consciente do risco. Isso possibilita um aprendizado seguro e eficiente mesmo sob condições complexas e imprevisíveis, tornando nossa abordagem mais robusta e amplamente aplicável do que os métodos anteriores baseados em modelo.

## III. Formulação do Problema

Abordamos a alocação dinâmica de recursos de rádio em uma RAN que atende a múltiplas fatias de rede heterogêneas. O desafio central é alocar recursos de forma eficiente, ao mesmo tempo em que se fornecem garantias probabilísticas robustas para o SLA de cada fatia em um ambiente não estacionário.

### A. Modelo do Sistema

Consideramos um sistema, conforme ilustrado na parte esquerda da Figura 1, composto por uma única estação-base que aloca um total de $C$ PRBs entre um conjunto de $L$ fatias de rede coexistentes, indexadas por $i \in \{1, \dots, L\}$. O sistema opera em passos de tempo discretos $t = 1, 2, \dots$, em que cada passo representa um período de decisão durante o qual uma alocação de recursos é mantida constante. Para alinhar com a especificação O-RAN, definimos que cada passo corresponde a 100 TTIs e que cada TTI tem duração de 1 ms. O ambiente é não estacionário, caracterizado pela chegada e partida de usuários da rede de acordo com um processo estocástico.

### B. Espaços de Estado e Ação

No início de cada passo de tempo $t$, o agente de controle observa um vetor de estado local $s_t^{(i)} \in \mathbb{R}^{d_i}$ para cada fatia $i$. Esse vetor contém Indicadores-Chave de Desempenho (KPIs) agregados, tais como o menor throughput dos usuários, o número de usuários, os tamanhos das filas e a Relação Sinal-Ruído (SNR). Para viabilizar um comportamento cooperativo, o estado de cada agente é enriquecido com uma representação de campo médio (*mean-field*) do estado global do sistema, incluindo estatísticas globais (por exemplo, uso total de PRBs das outras fatias, número total de UEs), $\bar{s}_t \in \mathbb{R}^{d_g}$, permitindo que o agente aprenda proativamente a correlação entre o congestionamento em todo o sistema e a necessidade de propostas conservadoras. A representação completa do estado para o agente $i$ é a concatenação $z_t^{(i)} = [s_t^{(i)}, \bar{s}_t]$.

Com base nessa observação, o agente seleciona uma ação conjunta $a_t = (a_t^{(1)}, \dots, a_t^{(L)})$, em que $a_t^{(i)} \in \{0, 1, \dots, C\}$ é o número de PRBs alocados à fatia $i$. A ação deve satisfazer a restrição de capacidade: $\sum_{i=1}^{L} a_t^{(i)} \le C$.

Após aplicar a ação $a_t$, o sistema realiza a transição, e uma métrica real de desempenho $y_t^{(i)}$ (por exemplo, o 5º percentil do throughput dos usuários) é observada para cada fatia.

### C. Objetivo: Um Processo de Decisão de Markov Restrito

O objetivo é encontrar uma política de controle distribuída, $\pi = (\pi^{(1)}, \dots, \pi^{(L)})$, que minimize o total de recursos de rádio alocados, garantindo, ao mesmo tempo, que o SLA de cada fatia seja atendido com alta probabilidade. Isso pode ser formulado como um Processo de Decisão de Markov Restrito (CMDP):

$$\min_{\pi} \lim_{T\to\infty} \frac{1}{T} \mathbb{E}_{\pi}\left[\sum_{t=1}^{T}\sum_{i=1}^{L} a_t^{(i)}\right]$$
$$\text{s.t. } \lim_{T\to\infty} \frac{1}{T} \mathbb{E}_{\pi}\left[\sum_{t=1}^{T} \mathbb{I}(y_t^{(i)} < \Lambda^{(i)})\right] \le \delta^{(i)};\ \forall i \in L \quad (1)$$

em que:

- $\mathbb{E}_{\pi}$ denota a esperança sobre todas as possíveis trajetórias do sistema sob a política conjunta $\pi$.
- $\Lambda^{(i)}$ é o valor mínimo aceitável de KPI (o limiar de SLA) para a fatia $i$.
- $\mathbb{I}(\cdot)$ é a função indicadora, que vale 1 se a condição for verdadeira (ocorre uma violação de SLA) e 0 caso contrário.
- $\delta^{(i)}$ é a taxa máxima tolerável de violação de longo prazo para a fatia $i$ (por exemplo, 1%).

A dinâmica de transição do sistema é desconhecida, o espaço de estados é amplo e contínuo, e o ambiente é não estacionário, o que torna este CMDP intratável para solução por métodos tradicionais.

## IV. Regressão Quantílica Online Segura

Propomos um sistema de controle distribuído e avesso ao risco, composto por $L$ agentes de aprendizado independentes. O núcleo da nossa solução é uma nova estrutura de aprendizado que integra Regressão Quantílica Online com mecanismos de segurança explícita, cooperação multiagente e gerenciamento adaptativo de risco.

### A. Modelo de Regressão Quantílica Online

Para cada fatia $i$, aprendemos uma função $f_{\tau_t^{(i)}}^{(i)}$ que, dado um estado enriquecido $z_t^{(i)}$ e uma ação $a_t^{(i)}$, prediz o quantil $\tau_t^{(i)}$-ésimo do KPI de desempenho $y_t^{(i)}$. A entrada do modelo é o vetor de características $x_t^{(i)} = [z_t^{(i)}, a_t^{(i)}/C]$.

O modelo $f_{\tau_t^{(i)}}^{(i)}$ é um aproximador de função kernelizado, definido por um conjunto de $B$ vetores de suporte (*landmarks*) $x_j$ e seus respectivos coeficientes $\alpha_j$, mantidos em um buffer de memória com orçamento fixo:

$$f_{\tau_t^{(i)}}^{(i)}(x) = \sum_{j=1}^{B} \alpha_j k(x, x_j),$$
$$\sigma_t^{(i)}(x) = 1 - \max_j k(x, x_j),$$

em que $k(\cdot, \cdot)$ é o kernel Matérn, escolhido por sua flexibilidade na modelagem de funções complexas e não lineares. O termo $\sigma_t^{(i)}(x)$ quantifica a incerteza da aproximação do modelo.

O agente aprimora a linha de base ao incorporar uma regra de aprendizado assimétrica e sensível a custo [13]. A atualização do coeficiente de um novo vetor de suporte, $\alpha_{new}^{(i)}$, é definida como:

$$\alpha_{UP}^{(i)} = \eta \cdot \tau_t^{(i)} \quad \text{se } y_t^{(i)} > f_{\tau_t^{(i)}}^{(i)}(x) \text{ (subpredição)}$$
$$\alpha_{OP}^{(i)} = \eta \cdot (\tau_t^{(i)} - 1) \quad \text{se } y_t^{(i)} \le f_{\tau_t^{(i)}}^{(i)}(x) \text{ (sobrepredição)}$$

em que $\eta$ é a taxa de aprendizado e $\tau_t^{(i)}$ é o quantil-alvo.

Em casos de falha crítica — em que uma ação predita como segura leva a uma violação de SLA — definimos ainda $\alpha_{CF}^{(i)} = \alpha_{OP}^{(i)} \cdot P_{penalty}$, em que $P_{penalty} > 1$ é um fator de amplificação de penalidade usado para enfatizar a gravidade de tais erros de predição.

O modelo é atualizado de forma online a cada passo. Quando uma nova observação $(x_t^{(i)}, y_t^{(i)})$ é recebida, um novo vetor de suporte é adicionado com seu coeficiente calculado a partir do gradiente da perda *pinball* [13]. Para manter o orçamento de memória $B$, os vetores de suporte menos informativos são descartados, priorizando aqueles que mais contribuem para o aprendizado. Ao longo do tempo, o modelo converge para uma fronteira de decisão sistematicamente mais avessa ao risco do que o quantil $\tau_t^{(i)}$, conforme demonstrado na Seção IV-D.

### B. Política de Controle Hierárquica

A política $\pi^{(i)}$ de cada agente seleciona uma ação por meio de um processo em múltiplas etapas, projetado para priorizar a segurança e, ao mesmo tempo, minimizar o consumo de recursos.

#### 1) Identificação do Conjunto Seguro

O agente primeiro identifica o conjunto de ações preditas para satisfazer a restrição de SLA com nível de confiança $(1 - \tau_t^{(i)})$. O conjunto de ações seguras é definido como:

$$\mathcal{A}_{safe,t}^{(i)} = \left\{ a \in \{0, \dots, C\} \mid f_{\tau_t^{(i)}}^{(i)}(z_t^{(i)}, a) \ge \Lambda^{(i)} \right\}$$

#### 2) Lógica de Seleção de Ação

Se $\mathcal{A}_{safe,t}^{(i)}$ for não vazio (Modo Confiante), o agente busca minimizar o uso de recursos mantendo a satisfação do SLA. Ele seleciona a ação com a maior pontuação ajustada pela eficiência:

$$a_t^{(i)} = \arg\max_{a \in \mathcal{A}_{safe,t}^{(i)}} \left( f_{\tau_t^{(i)}}^{(i)}(z_t^{(i)}, a) - \gamma \cdot (a/C) \right)$$

em que $\gamma$ é uma penalidade de custo que promove a eficiência de recursos.

Se $\mathcal{A}_{safe,t}^{(i)}$ estiver vazio (Modo Incerto), o agente recorre a uma estratégia exploratória hierárquica: com probabilidade $\epsilon$, ele amostra uma ação exploratória baseada em Softmax:

$$a_t^{(i)} \sim P(a) \propto \exp\left( \frac{f_{\tau_t^{(i)}}^{(i)}(z_t^{(i)}, a)}{T_{softmax}} \right),$$

em que $T_{softmax}$ é a temperatura que controla a nitidez da exploração. Com probabilidade $1 - \epsilon$, ele invoca uma heurística *Safety-First* de $k$-NN, buscando os $k$ estados passados mais próximos e selecionando uma ação conhecida como bem-sucedida (isto é, $y \ge \Lambda^{(i)}$):

$$a_t^{(i)} = \arg\max_{a \in \mathcal{N}_{K_i,t}^{(i)}} \left( f_{\tau_t^{(i)}}^{(i)}(z_t^{(i)}, a) \right),$$

em que $\mathcal{N}_{K_i,t}^{(i)}$ é o conjunto de ações executadas nos $k$ estados vizinhos mais próximos.

#### 3) Ajuste de Recursos

Para aumentar a confiabilidade sob incerteza, o agente ajusta cada ação por uma margem de segurança:

$$a_t^{(i)} \leftarrow a_t^{(i)} + \text{round}(\sigma_t^{(i)} \cdot M_t^{(i)}),$$

em que $\sigma_t^{(i)}$ é a incerteza do modelo e $M_t^{(i)}$ controla a escala da margem. Margens maiores são aplicadas em casos de alta incerteza para evitar violações de SLA.

Se a soma dos recursos solicitados exceder a capacidade do sistema ($\sum_i a_t^{(i)} > C$), um operador de projeção centralizado $\Pi$ reduz proporcionalmente o vetor de alocação para impor a viabilidade. Se a ação de uma fatia for reduzida devido a essa projeção, uma atualização virtual é disparada: um vetor de suporte representando a ação original (pré-projeção) é adicionado à memória do aprendiz com um pequeno peso negativo. Isso penaliza propostas excessivamente agressivas e orienta o agente a evitar decisões que exijam correção frequente.

### C. Adaptação Dinâmica do Quantil e da Margem

Para adaptar seu conservadorismo, o agente ajusta dinamicamente o quantil $\tau_t^{(i)}$ e a escala de margem $M_t^{(i)}$ por fatia, por meio de um controlador heurístico que acompanha as violações recentes de SLA. Quando a taxa de violação aumenta, o controlador reduz $\tau_t^{(i)}$ para incentivar predições mais pessimistas e aumenta $M_t^{(i)}$ para adicionar uma margem de segurança maior em termos de RBs. Inversamente, quando o desempenho de SLA está quase perfeito, ele eleva cautelosamente $\tau_t^{(i)}$ e reduz $M_t^{(i)}$, melhorando assim a eficiência de recursos sem comprometer a confiabilidade.

### D. Análise de Convergência da Regra de Aprendizado Avessa ao Risco

A convergência para um valor numérico fixo é intratável devido à não estacionariedade do ambiente e ao comportamento adaptativo do agente. No entanto, provamos que o agente converge consistentemente para uma fronteira de decisão mais avessa ao risco do que o quantil-$\tau$ de linha de base. Nosso agente baseia-se em regressão quantílica online, que atualiza as predições usando o gradiente da perda *pinball*. A formulação padrão garante convergência para o verdadeiro quantil-$\tau$ da distribuição condicional $\mathbb{P}(y|x)$ [13], de modo que a subpredição ocorra com probabilidade $\tau$ e a sobrepredição com $1 - \tau$. A convergência é alcançada quando a atualização esperada se torna zero: $\mathbb{E}[\alpha_{new}] = 0$. Com nossa regra de atualização sensível a custo, a condição de equilíbrio torna-se:

$$\mathbb{P}(y > f(x)) \cdot \alpha_{UP} + \mathbb{P}(\Lambda \le y \le f(x)) \cdot \alpha_{OP} + \mathbb{P}(y < \Lambda) \cdot \alpha_{CF} = 0$$

Substituímos os valores de atualização e rearranjamos essa equação para mostrar o equilíbrio sobre a predição:

$$\underbrace{\mathbb{P}(y > f(x)) \cdot \tau}_{\text{Força Total para Cima}} = \underbrace{\mathbb{P}(\Lambda \le y \le f(x)) \cdot (1-\tau)}_{\text{Força Normal para Baixo}} + \underbrace{\mathbb{P}(y < \Lambda) \cdot (1-\tau) \cdot P_{penalty}}_{\text{Força Amplificada para Baixo}}$$

Para manter o equilíbrio sob essa força descendente mais intensa, o modelo deve aumentar a probabilidade de subpredição, $\mathbb{P}(y > f(x))$. A única maneira de alcançar isso é reduzir a predição $f(x)$, deslocando assim a fronteira de decisão em uma direção mais conservadora.

Isso estabelece que a regra de aprendizado sensível a custo induz convergência para uma fronteira de decisão mais pessimista (isto é, mais segura) do que o quantil-$\tau$ nominal. Embora a taxa final exata de satisfação de SLA dependa da dinâmica estocástica do ambiente, o processo de aprendizado é comprovadamente enviesado para maior segurança. Assim, nosso método fornece um mecanismo fundamentado para adaptação avessa ao risco, sem exigir restrições explícitas ou correção *post-hoc*.

**Algoritmo 1 — Regressão Quantílica Online Segura**

```text
Entrada: ε, γ, C, K, η
Inicializar: {f₁⁽ⁱ⁾, τ₁⁽ⁱ⁾, M₁⁽ⁱ⁾, σ₁⁽ⁱ⁾}ᵢ₌₁ᴸ, B = ∅
para t = 1, …, T faça:
  para i = 1, …, L faça:
    A(safe,t)⁽ⁱ⁾ = { a ∈ {0,…,C} | fₜ⁽ⁱ⁾(zₜ⁽ⁱ⁾,a) ≥ Λ⁽ⁱ⁾ }
    se A(safe,t)⁽ⁱ⁾ ≠ ∅ então:
      U(a) ← fₜ⁽ⁱ⁾(zₜ⁽ⁱ⁾,a) − γ·a/C
      a(base,t)⁽ⁱ⁾ ← arg max_{a∈A(safe)} U(a)
    senão:
      se random() < ε então:
        a(base,t)⁽ⁱ⁾ ∼ exp(fₜ⁽ⁱ⁾(zₜ⁽ⁱ⁾,a)/T_softmax)
      senão:
        a(base,t)⁽ⁱ⁾ ← arg max_{a∈N(Ki,t)} (fₜ⁽ⁱ⁾(zₜ⁽ⁱ⁾,a))
    a(final,t)⁽ⁱ⁾ ← a(base,t)⁽ⁱ⁾ + round(σₜ⁽ⁱ⁾·Mₜ⁽ⁱ⁾)
  se Σᵢ a(final,t)⁽ⁱ⁾ > C então:
    a(final,t) ← Π(a(final,t))  ▷ Projetar para o conjunto viável
  Executar a(final,t), observar resultado yₜ e próximo estado sₜ
  para i = 1, …, L faça:
    Atualizar fₜ₊₁⁽ⁱ⁾ com (zₜ⁽ⁱ⁾, a(final,t)⁽ⁱ⁾, yₜ⁽ⁱ⁾)
    Aplicar atualização de penalidade de ajuste
    Atualizar τₜ₊₁⁽ⁱ⁾, Mₜ₊₁⁽ⁱ⁾
```



## V. Avaliação Experimental e Análise

Para avaliar a eficácia do agente de Regressão Quantílica Segura (SQR) proposto, realizamos uma simulação abrangente em nível de sistema, projetada para emular a dinâmica de uma implantação O-RAN realista, na qual o modelo é atualizado a cada 100 *slots* e cada atualização leva menos de 100 ms. Nossa simulação modela uma única estação-base macrocelular com raio de cobertura de 2 km, operando na frequência portadora de 2 GHz. A propagação de rádio segue o modelo 3GPP de perda de percurso urbano para macrocélulas [14], combinado com o perfil de desvanecimento Pedestrian A [15], para capturar variações realistas de sinal induzidas pela mobilidade. A estação-base atende a um conjunto de usuários móveis cujas posições espaciais e trajetórias são geradas aleatoriamente por episódio. As chegadas de usuários seguem um processo de Poisson, e a duração de cada conexão é distribuída exponencialmente para simular durações de sessão variáveis. O sistema suporta três fatias eMBB, cada uma definida por configurações distintas de SLA em termos de limiares mínimos de throughput. Cada UE é atribuído aleatoriamente a uma das fatias e gerenciado pelo agente correspondente. Mantemos uma memória de vetores de suporte com 2000 entradas e faixa de quantis de [0,01; 0,05]. A Tabela I resume a lista completa de hiperparâmetros da simulação e do agente.

**Tabela I – Parâmetros de Simulação**

| Parâmetros do Canal de Rádio |  | Modelo de Tráfego |  |
|---|---|---|---|
| Modelo de propagação | Macrocelular urbano | Chegadas de UEs | Processo de Poisson com 2 usuários/min |
| Modelo de desvanecimento | Pedestrian A | Tempo de conexão | Distribuição exponencial com média de 60 segundos |
| Potência transmitida | 30 dBm | SLAs $\Lambda$ | 0,5 Mbps/UE, 1 Mbps/UE, 1,5 Mbps/UE |
| Ganho de antena da BS | 15 dBi | **Parâmetros SQR** |  |
| Alcance da célula | 2 km | $P_{penalty}$ | 2 |
| Ruído | 9 dB | Aleatoriedade $\epsilon$ | 0,1 |
| Interferência mais ruído | -110 dBm | Taxa de aprendizado $\eta$ | 0,1 |
| Frequência portadora | 2 GHz | $K$ vizinhos mais próximos | 10 |

Avaliamos o agente SQR sob duas configurações — com ($\gamma = 1$) e sem ($\gamma = 0$) o mecanismo de penalidade sensível a custo — e comparamos ambas as variantes com o KBRL [12], uma linha de base de aprendizado por reforço baseado em modelo previamente estudada para controle O-RAN. As avaliações são conduzidas ao longo de múltiplas execuções para garantir confiabilidade estatística, e todas as métricas são agregadas nas três fatias para permitir comparação consistente entre métodos.

A Figura 2 mostra as taxas médias de violação de SLA ao longo do tempo. Embora todos os métodos eventualmente convirjam para satisfazer a meta de 99% de SLA, diferenças importantes surgem durante os estágios iniciais de aprendizado.

> **Figura 2 – Taxa de Violação de SLA** *[inserir imagem original do artigo]*

A Figura 3 apresenta as tendências cumulativas de violação, em que ambas as variantes SQR exibem curvas significativamente mais planas do que o KBRL, particularmente durante a fase inicial. Isso demonstra convergência mais rápida e eficiência amostral superior — vantagens críticas em ambientes online, nos quais decisões iniciais ruins podem levar à degradação prolongada da experiência do usuário. Notavelmente, o KBRL é avaliado sob restrições de SLA por fatia, enquanto nosso método impõe o SLA em nível por UE, resultando em garantias mais rigorosas e granulares. Consequentemente, o KBRL tende a incorrer em mais violações devido à sua granularidade de imposição mais grosseira. Além disso, ao contrário de abordagens que se baseiam em suposições monolíticas sobre o comportamento da rede, nosso modelo aprende diretamente a partir de resultados detalhados da rede, permitindo predição de risco de SLA mais precisa e controle mais confiável.

Um desafio central em sistemas de aprendizado online é a memória limitada. Em nossa configuração, a memória de vetores de suporte é limitada a 2000 entradas. Apesar dessa restrição, o agente SQR continua mantendo predições de alta qualidade durante todo o treinamento. Isso é alcançado por meio de uma estratégia adaptativa de poda de memória que descarta amostras de baixo impacto ou redundantes, maximizando assim o valor de aprendizado de cada vetor de suporte armazenado. O agente aprende efetivamente quais experiências são mais valiosas para tomadas de decisão futuras, demonstrando robustez sob capacidade limitada.

A comparação das duas variantes SQR revela um *trade-off* estratégico introduzido pela penalidade sensível a custo. A variante com penalidades incorre em ligeiramente mais violações de SLA — tipicamente na faixa de 0,003–0,008% — mas consome menos recursos de rádio. Esse comportamento é esperado: o agente é incentivado a se aproximar mais do limite do SLA em busca de maior eficiência de recursos. O mecanismo de penalidade leva o agente a tomar decisões mais ajustadas e agressivas, aumentando ocasionalmente o risco de pequenas violações de SLA, mas proporcionando, em troca, eficiência significativamente maior. É importante ressaltar que essas violações são raras e de pequena magnitude, o que sugere que a política baseada em penalidades está bem calibrada para implantação prática.

A Figura 4 apresenta o número médio de RBs utilizados ao longo do tempo. Ambos os agentes SQR alcançam consumo de RBs substancialmente menor do que o KBRL, com o SQR com penalidade apresentando a maior eficiência — economizando consistentemente mais de 60% no uso de RBs. Essa economia de recursos é particularmente importante em ambientes multifatia, em que os recursos espectrais são limitados e precisam ser compartilhados entre demandas concorrentes de serviço. Os resultados sugerem que nossa abordagem não apenas atende às metas de nível de serviço, mas também promove uma utilização globalmente eficiente dos recursos.

> **Figura 3 – Violação Acumulada de SLA** *[inserir imagem original do artigo]*
>
> **Figura 4 – Alocação de Recursos** *[inserir imagem original do artigo]*

Outra observação importante é a capacidade de generalização da estrutura SQR. Ao contrário do KBRL, que depende de uma relação monotônica entre ações e resultados, nosso método não impõe tais suposições. Isso permite que o SQR permaneça eficaz mesmo sob efeitos não monotônicos da camada física, como os causados por interferência intercelular, *jitter* de escalonamento ou padrões abruptos de mobilidade. Essas complexidades do mundo real frequentemente levam a comportamentos contraintuitivos, nos quais recursos adicionais podem não resultar em melhor desempenho. A capacidade de nosso método de modelar tais fenômenos o torna mais adequado para implantação em sistemas O-RAN de produção.

## VI. Conclusão e Trabalhos Futuros

Este artigo apresenta um agente de aprendizado por reforço baseado em modelo para O-RAN que alcança aprendizado online seguro e eficiente. Ao alavancar a regressão quantílica online, o agente modela explicitamente a fronteira de risco de SLA e aplica uma regra de aprendizado sensível a custo que promove tanto a segurança quanto a eficiência de recursos — sem depender de suposições fortes sobre a dinâmica da rede. Os experimentos mostram que o método proposto reduz as violações acumuladas de SLA em 60% e diminui o consumo de recursos de rádio em 20% em comparação com as linhas de base, alcançando um forte equilíbrio entre segurança e eficiência. Esses resultados destacam o potencial do agente SQR para viabilizar o gerenciamento inteligente e automatizado de fatias em cenários O-RAN práticos.

Como trabalhos futuros, planejamos validar nossa estrutura em *testbeds* O-RAN reais para avaliar sua robustez sob condições práticas, como latência de sinalização, restrições de hardware e variabilidade de tráfego. Pretendemos prototipar o agente em *testbeds* 5G de larga escala, como Colosseum [16] e Arena [17], para avaliar a transferência da simulação para a realidade, a responsividade e a segurança. Além disso, estenderemos o agente para lidar com os três tipos de fatia 5G — eMBB, URLLC e mMTC — para testar sua generalização e desenvolver mecanismos adaptativos de coordenação inter-fatia em ambientes O-RAN multisseriço.

## Agradecimentos

Este trabalho foi parcialmente apoiado pela NICT, concessão nº 09101 “P&D para Otimização de Sistemas Móveis 6G com Integração de IA Cross-Layer/Multi-Domínio”, pelo Programa de Cooperação Social com a KYOCERA e pela concessão JST ASPIRE nº JPMJAP2323, Japão.

## Referências

[1] Wei Jiang, Bin Han, Mohammad Asif Habibi, and Hans Dieter Schotten, “The road towards 6g: A comprehensive survey,” *IEEE Open Journal of the Communications Society*, vol. 2, pp. 334–366, 2021.

[2] Ibrahim Afolabi, Tarik Taleb, Konstantinos Samdanis, Adlen Ksentini, and Hannu Flinck, “Network slicing and softwarization: A survey on principles, enabling technologies, and solutions,” *IEEE Communications Surveys & Tutorials*, vol. 20, no. 3, pp. 2429–2453, 2018.

[3] Michele Polese, Leonardo Bonati, Salvatore D’oro, Stefano Basagni, and Tommaso Melodia, “Understanding o-ran: Architecture, interfaces, algorithms, security, and research challenges,” *IEEE Communications Surveys & Tutorials*, vol. 25, no. 2, pp. 1376–1411, 2023.

[4] Aerman Tuerxun and Akihiro Nakao, “Energy-efficient ldpc acceleration in open radio access networks using drl and qrf,” in *2025 IEEE Wireless Communications and Networking Conference (WCNC)*. IEEE, 2025, pp. 1–6.

[5] Aerman Tuerxun and Akihiro Nakao, “Sora: Energy-efficient resource allocation in open radio access network with online learning,” *IEEE Access*, 2025.

[6] Noe M Yungaicela-Naula, Vishal Sharma, and Sandra Scott-Hayward, “Rslaq–a robust sla-driven 6g o-ran qos xapp using deep reinforcement learning,” *arXiv preprint arXiv:2504.09187*, 2025.

[7] Yuxiu Hua, Rongpeng Li, Zhifeng Zhao, Xianfu Chen, and Honggang Zhang, “Gan-powered deep distributional reinforcement learning for resource management in network slicing,” *IEEE Journal on Selected Areas in Communications*, vol. 38, no. 2, pp. 334–349, 2019.

[8] Yi Jia, Cheng Zhang, Nan Li, Yongming Huang, and Tony QS Quek, “Hierarchical intelligence enabled joint ran slicing and mac scheduling for sla guarantee,” *IEEE Transactions on Communications*, 2024.

[9] Mingrui Yin, Yang Deng, Ahan Kak, Nakjung Choi, and Tao Han, “Psaslicing: Perpetual sla-aware reinforcement learning for o-ran slice management,” in *GLOBECOM 2024-2024 IEEE Global Communications Conference*. IEEE, 2024, pp. 4534–4539.

[10] Muhammad Sulaiman, Mahdieh Ahmadi, Bo Sun, Mohammad A Salahuddin, Raouf Boutaba, and Aladdin Saleh, “Microopt: Model-driven slice resource optimization in 5g and beyond networks,” *IEEE Transactions on Network and Service Management*, 2025.

[11] Farhad Rezazadeh, Hatim Chergui, and Josep Mangues-Bafalluy, “Explanation-guided deep reinforcement learning for trustworthy 6g ran slicing,” in *2023 IEEE International Conference on Communications Workshops (ICC Workshops)*. IEEE, 2023, pp. 1026–1031.

[12] Juan J Alcaraz, Fernando Losilla, Andrea Zanella, and Michele Zorzi, “Model-based reinforcement learning with kernels for resource allocation in ran slices,” *IEEE Transactions on Wireless Communications*, vol. 22, no. 1, pp. 486–501, 2022.

[13] Roger Koenker and Gilbert Bassett Jr, “Regression quantiles,” *Econometrica: Journal of the Econometric Society*, pp. 33–50, 1978.

[14] Evolved Universal Terrestrial Radio Access (E-UTRA), “Radio frequency (rf) system scenarios,” document 36.942, Version 15.0.0, 3GPP, 3rd Generation Partnership Project (3GPP), Jun. 2018.

[15] Evolved Universal Terrestrial Radio Access (E-UTRA), “Base station (bs) radio transmission and reception,” document 36.104, Version 15.15.0, 3GPP, 3rd Generation Partnership Project (3GPP), Apr. 2022.

[16] Leonardo Bonati, Pedram Johari, Polese, et al., “Colosseum: Large-scale wireless experimentation through hardware-in-the-loop network emulation,” in *2021 IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN)*. IEEE, 2021, pp. 105–113.

[17] Lorenzo Bertizzolo, Leonardo Bonati, Emrecan Demirors, and Tommaso Melodia, “Arena: A 64-antenna sdr-based ceiling grid testbed for sub-6 ghz radio spectrum research,” in *Proceedings of the 13th International Workshop on Wireless Network Testbeds, Experimental Evaluation & Characterization*, 2019, pp. 5–12.

