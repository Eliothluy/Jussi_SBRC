# Plano técnico de pesquisa e implementação — Reprodução do SQR e evolução para coordenação inter-slice em O-RAN

**Versão:** 1.0 — 09/10/2026  
**Público-alvo:** pesquisador responsável pela implementação, simulação, coleta e análise dos resultados; dupla responsável pela formulação científica.  
**Situação:** plano de execução e especificação inicial; **não** representa código já implementado nem resultados experimentais.  
**Artigo-base:** A. Tuerxun e A. Nakao, *Safe RAN Slicing in O-RAN: Minimizing SLA Violations via Model-Based Reinforcement Learning*, IEEE CCNC 2026, DOI: [10.1109/CCNC65079.2026.11366589](https://doi.org/10.1109/CCNC65079.2026.11366589) (6 páginas).  
**Ambientes pretendidos:** Python (prototipagem dos agentes e experimentos) e ns-3 com 5G-LENA (validação com PHY/MAC/tráfego). A integração com RIC/xApp e E2 é uma etapa separada, não pressuposta.

> **Regra de rastreabilidade.** Neste documento, `[ARTIGO]` significa especificação explicitamente descrita no PDF; `[DERIVADO]` significa consequência matemática/operacional dessa especificação; `[PROPOSTA]` é desenho sugerido para o novo estudo; `[PENDENTE]` é lacuna de implementação ou informação a confirmar. **Não apresentar escolhas propostas como configurações originais dos autores.**

---

## 0. Resumo executivo: o que deve ser entregue

O projeto deve seguir uma sequência em que **cada etapa tem resultados verificáveis antes da próxima**:

1. **Auditoria de especificação e ambiente** — catalogar o que o artigo divulga e o que falta; congelar versões, interfaces, unidades, sementes e formato dos logs.
2. **Reprodução algorítmica em Python** — implementar SQR sem custo (`γ=0`), SQR com custo (`γ=1`), avaliação de risco, kernel Matérn, memória limitada, exploração, margem de segurança, coordenação/projeção original; reproduzir o cenário com **3 slices eMBB** e comparar com KBRL.
3. **Estudo dos quatro regimes de carga** — carga baixa, normal, estressada e congestionada, incluindo **transições entre regimes**, mantendo inicialmente 3 eMBB para isolar o efeito da carga.
4. **Generalização para slices heterogêneas** — primeiro eMBB + URLLC (throughput e latência), depois eMBB + URLLC + mMTC (métrica mMTC definida antes da execução). Avaliar a adaptação do SQR original, sem atribuir-lhe capacidades não implementadas.
5. **Evolução científica proposta (nome provisório `SQR-JC`)** — integrar um **coordenador conjunto, consciente do risco de SLA**, substituindo a projeção meramente proporcional quando houver conflito de capacidade. Comparar na **mesma carga, topologia, canal e métricas**.
6. **Validação em ns-3/5G-LENA** — transportar controladores e cenários para uma rede detalhada; distinguir um controle direto de PRBs de uma simples configuração estática de bandas/BWPs. Medir fidelidade e latência da decisão.
7. **Resultados do artigo** — figuras equivalentes às Fig. 2, 3 e 4 do SQR, tabelas por slice/cenário, curvas de recuperação após mudanças, análise estatística, ablações e limitações.

**Decisão de escopo:** **DRL não é obrigatória**. O eixo inicial é aperfeiçoar o controle do SQR. DDQN/RSLAQ podem entrar como baselines somente se forem implementados e avaliados de forma equivalente, sem comprometer a entrega principal.

### Definição do problema em uma frase

O SQR original aprende a prever risco **por slice** e corrige excesso de propostas de PRBs com projeção global; a pesquisa investigará **quando isso deixa de preservar SLAs heterogêneos** e se uma **coordenação conjunta informada pelo risco** reduz violações e melhora o uso de PRBs em condições dinâmicas de carga.

### Critério central de validade

Não basta obter menos violações ao custo de reservar quase todos os PRBs; não basta usar menos PRBs degradando indiscriminadamente os SLAs. Reportar **o trade-off risco × consumo**, incluindo casos em que é **fisicamente impossível** satisfazer simultaneamente todos os SLAs.

---

## 1. Leitura operacional e limites do artigo-base

### 1.1 O problema formal que o artigo resolve [ARTIGO, Seção III]

Uma estação-base reparte um total de `C` PRBs entre `L` slices. A cada decisão `t`, o agente `i` observa um estado local `s_t^(i)` combinado a uma visão agregada global `s̄_t`:

```text
z_t^(i) = concat(s_t^(i), s̄_t)
a_t = (a_t^(1), ..., a_t^(L))
a_t^(i) ∈ {0, 1, ..., C}
soma_i a_t^(i) ≤ C
```

Os estados locais incluem **KPIs agregados**, como menor throughput dos UEs, quantidade de usuários, filas e SNR. A informação global pode conter consumo de PRBs das demais slices e quantidade total de UEs. A composição **exata e a normalização de cada componente** não são completamente definidas no artigo.

O objetivo formal é minimizar a utilização média de PRBs, sujeito à taxa de violação de cada slice não exceder um limite `δ_i`:

```text
min_π lim_(T→∞) (1/T) E_π [ soma_t soma_i a_t^(i) ]
sujeito a:
  lim_(T→∞) (1/T) E_π [ soma_t I(y_t^(i) < Λ_i) ] ≤ δ_i, para cada i
```

O texto fornece como exemplo `δ_i = 1%`. O exemplo de desempenho `y_t^(i)` é o **5º percentil de throughput dos UEs**. **A formulação escrita usa o desempenho agregado por slice**, mas a Seção V afirma que a solução impõe SLA em granularidade *per-UE*, enquanto o KBRL está em granularidade *per-slice*. Essa diferença **precisa ser tratada explicitamente** na comparação: registrar resultados nos dois níveis e definir uma métrica principal comum.

### 1.2 O que foi realmente avaliado [ARTIGO, Seção V]

- **Três slices eMBB**, com três limiares mínimos diferentes de throughput; **não** uma slice eMBB, uma URLLC e uma mMTC.
- Uma estação-base, posicionamento e trajetórias dos UEs aleatórios por episódio.
- Chegadas e partidas estocásticas de UEs.
- Dois SQR: **`SQR w/o Cost (γ=0)`** e **`SQR w/ Cost (γ=1)`**.
- Comparação com KBRL (Alcaraz et al.).
- Gráficos principais: **Fig. 2** (taxa de violações ao longo da aprendizagem), **Fig. 3** (violações acumuladas), **Fig. 4** (distribuição do número de RBs consumidos).
- Memória de vetores de suporte limitada a 2.000 entradas.
- O artigo usa uma **simulação de sistema** e uma **arquitetura conceitual de O-RAN/RIC**. Não identifica inequivocamente o simulador/fonte/código, nem demonstra uma xApp integrada a E2AP/E2SM-RC.

**Cuidado com as porcentagens divulgadas:** a conclusão relata 60% menos violações acumuladas e 20% menos consumo de recursos que baselines, mas a discussão da Fig. 4 menciona economia de RBs superior a 60%. Sem uma definição inequívoca do denominador e da agregação, **não fixar essas porcentagens como critérios de aceitação da reprodução**. Recalcular os ganhos a partir de séries obtidas no nosso ambiente.

### 1.3 Os dois mecanismos de “custo” não são o mesmo

1. `γ` aparece **na seleção da ação segura** e penaliza uso de PRBs: `f_τ(z,a) − γ(a/C)`. **Diferencia as duas variantes experimentais:** `γ=0` e `γ=1`.
2. `P_penalty=2` aparece **na atualização do regressor após uma falha crítica** (ação prevista como segura que viola SLA). Essa penalização assimétrica de erro é descrita para o agente; **não** equivale a `γ`.

**Não implementar `SQR w/o Cost` desativando a penalização de falha crítica**. Isso faria um método diferente do comparado no artigo.

---

## 2. Parâmetros do cenário original: transcrição + rastreabilidade

### 2.1 Parâmetros publicados [ARTIGO]

| Categoria | Parâmetro | Valor publicado | Fonte / observação |
|---|---|---|---|
| Topologia | Estações-base | **1** macrocélula | Seções III-A e V |
| Slices | Quantidade | **3** | Seção V |
| Slices | Tipos | **3 eMBB** | Seção V; três limiares de throughput |
| SLA | Throughput mínimo por UE | **0,5; 1,0; 1,5 Mbps/UE** | Tabela I |
| SLA | Alvo de taxa de violação | **1% (99% de atendimento)** | Seção III-C, Fig. 2 e texto da Seção V |
| Espaço de ação | PRBs por slice | `a_i ∈ {0, …, C}` | Seção III-B |
| Restrição física | Soma dos PRBs | `Σ_i a_i ≤ C` | Seção III-B |
| Tempo | TTI de referência | **1 ms** | Seção III-A |
| Tempo | Período de decisão | **100 TTIs = 100 ms** | Seção III-A |
| Rádio | Frequência de portadora | **2 GHz** | Tabela I |
| Rádio | Alcance da célula | **2 km** | Tabela I |
| Rádio | Potência de transmissão | **30 dBm** | Tabela I |
| Rádio | Ganho de antena da BS | **15 dBi** | Tabela I |
| Rádio | Figura de ruído (interpretar/confirmar) | **9 dB**, coluna `Noise` | Tabela I; unidade dB |
| Rádio | Campo “Inference [sic] plus noise” | **−110 dBm** | Tabela I; texto impresso é ambíguo (provável interferência + ruído); **não equiparar automaticamente a figura de ruído** |
| Canal | Propagação | **Macro cell urban** (ref. 3GPP [14]) | Tabela I |
| Canal | Fading | **Pedestrian A** | Tabela I |
| Tráfego | Chegadas de UEs | **Poisson, 2 usuários/minuto** | Tabela I; não fica inequívoco se a taxa é global ou por slice |
| Tráfego | Tempo de conexão | **Exponencial, média de 60 s** | Tabela I |
| Aprendizagem | Tipo de regressão | **Online Quantile Regression** com kernel **Matérn** | Seção IV-A |
| Aprendizagem | Intervalo de quantil adaptativo `τ` | **[0,01; 0,05]** | Seção V |
| Aprendizagem | Taxa `η` | **0,1** | Tabela I |
| Aprendizagem | Amplificador `P_penalty` | **2** | Tabela I |
| Controle | Probabilidade de exploração `ε` | **0,1** | Tabela I |
| Controle | Número de vizinhos `K` | **10** | Tabela I |
| Controle | Custo de PRBs `γ` | **0** (sem custo) / **1** (com custo) | Seção V |
| Memória | Orçamento de vetores de suporte `B` | **2.000 entradas** | Seção V |
| Avaliação | Métodos centrais | **KBRL; SQR w/o Cost; SQR w/ Cost** | Seção V |
| Avaliação | Atualização | A cada **100 slots/TTIs segundo o texto**; tempo de atualização inferior a **100 ms** é alegado | Seção V; em NR a duração de *slot* depende da numerologia, não confundir com 100 ms |

**Importante:** uma configuração de **100 PRBs** é apenas **hipótese experimental** comum para exemplos; o artigo **não informa o valor de `C` ou a largura de banda**. Portanto, não colocar `C=100` na “configuração fiel ao artigo” antes de verificar com os autores ou documentar como suposição.

### 2.2 Parâmetros indispensáveis não suficientemente documentados [PENDENTE]

| Item | Por que é necessário | Procedimento |
|---|---|---|
| `C`, largura de banda, SCS, numerologia, duplexação e PRB disponíveis | Define espaço de ação e capacidade real | Solicitar aos autores; caso contrário, derivar `C` da configuração NR **escolhida por nós** e registrar divergência |
| Esquema exato de tráfego, direção DL/UL, protocolo, tamanhos de pacotes | Throughput/latência e filas dependem disso | Definir versão experimental e reportar como assumida |
| Número inicial de UEs, distribuição por slice e taxa Poisson global ou por slice | Mudanças podem alterar radicalmente a carga | Solicitar; manter duas interpretações em piloto de sensibilidade se necessário |
| Total de episódios, número de passos, duração, aquecimento e sementes | Reprodutibilidade, convergência e intervalos de confiança | Especificar e versionar; a escala no eixo da figura **não substitui especificação** |
| Modelo detalhado de canal, perdas, parâmetros de Pedestrian A, mobilidade e interferência | Compatibilidade com ns-3/5G-LENA | Registrar todos os defaults do simulador e diferenças em relação ao artigo |
| Features exatas de `s_i` e `s̄`, suas unidades e normalização | Kernel pode se tornar dominado por uma feature | Definir esquema/normalização sem vazamento de futuro |
| Kernel Matérn: `ν`, `ℓ`/larguras por dimensão, escala/amplitude | Muda significativamente a predição | Seleção documentada via piloto, sem escolher olhando o teste final |
| Definição de `σ` quando a memória está vazia e normalização de `k` | `σ = 1 − max k` exige kernel normalizado no intervalo apropriado | Fixar `σ=1` para memória vazia e verificar `k(x,x)=1` na implementação proposta |
| Regra exata de *pruning* dos 2.000 vetores | Determina adaptação, retenção e custo | Implementar política explícita, registrar escolha e testar ablação |
| Inicialização e adaptação exata de `τ_t` e `M_t` | O texto descreve comportamento, mas não fornece todas as constantes/regra operacional | Solicitar; criar versões próprias claramente nomeadas |
| Temperatura `T_softmax` | Necessária no modo incerto | Fixar e justificar como hipótese experimental |
| Distância e representação usadas no KNN | Escolha de ações anteriores no modo incerto | Fixar e testar |
| Detalhamento da “atualização virtual negativa” após projeção | Não há coeficiente numérico explícito | Registrar regra implementada e executar ablação |
| Regra de desempate e arredondamento, cotas mínimas de PRB | Afeta somas e SLA | Definir deterministicamente com preservação de `Σa_i ≤ C` |
| Definição exata de violação per-UE versus por slice | Comparação injusta sem denominador comum | Instrumentar ambas e escolher uma métrica primária explícita |
| Software/simulador original, código e commit | Reproducibilidade forte | Procurar artefatos; contatar autores se necessário |

**Critério editorial:** sem o código dos autores ou as decisões da tabela acima, o resultado será **reimplementação do SQR a partir do artigo**, não uma reprodução bit a bit. Isso deve aparecer no método científico.

---

## 3. Implementar o SQR fiel à lógica publicada — primeiro em Python

### 3.1 Interfaces mínimas

O código precisa separar o **ambiente** (que gera observações e consequências) do **controlador** (que decide os PRBs):

```python
observation = env.reset(seed=seed)
for step in range(n_steps):
    allocation = controller.act(observation)          # vetor de PRBs por slice
    observation_next, measurements, done = env.step(allocation)
    controller.update(observation, allocation, measurements)
    logger.write(observation, allocation, measurements)
    observation = observation_next
    if done:
        break
```

**Esboço de interface, não código executável do artigo.** O ambiente deve oferecer medidas observáveis **antes** da ação; o KPI realizado usado no update só fica disponível **depois** do período de alocação.

### 3.2 Dados de entrada e modelos individuais por slice [ARTIGO]

Para cada slice `i`, construir:

```text
x_t^(i) = concat(z_t^(i), a_t^(i) / C)
z_t^(i) = concat(s_t^(i), s̄_t)
```

Regressor quantílico baseado em expansão por kernels:

```text
f_τ^(i)(x) = soma_(j=1..B) α_j k(x, x_j)
σ_t^(i)(x) = 1 − max_j k(x, x_j)
```

- `k` é kernel **Matérn**, não função de valor Q.
- Cada agente/slice mantém memória com **até 2.000 vetores** e coeficientes associados.
- `f_τ` estima um quantil condicional do KPI para o estado e a **ação candidata**.
- A expressão de `σ` mede **novidade/similaridade com a memória**, e **não** um intervalo probabilístico calibrado de incerteza. Evitar chamar `1−σ` de “probabilidade real de o SLA ser atendido”.
- O uso de `γ` **não altera a função de kernel**: altera a decisão dentre ações previstas seguras.

**Ponto de engenharia:** todos os atributos do estado precisam ser normalizados com escalas explícitas (contagem de usuários, SNR em dB, filas em bytes/pacotes, throughput em Mbps etc.). Deve ser possível salvar e recarregar o normalizador.

**Não substituir inadvertidamente** o método por `sklearn.QuantileRegressor` (linear) ou por um `GaussianProcessRegressor` genérico. Essas APIs não reproduzem automaticamente o **regressor quantílico kernelizado online com orçamento e atualização de coeficientes** descrito.

### 3.3 Regra de aprendizado [ARTIGO, Seção IV-A]

Com taxa de aprendizagem `η` e alvo `τ_t`:

```text
se y_real > f_τ(x):             α_novo = η · τ_t      # subpredição
senão:                          α_novo = η · (τ_t−1)  # sobrepredição
se ação prevista segura violou SLA:
                                α_novo = [η · (τ_t−1)] · P_penalty
```

- O terceiro caso (**falha crítica**) usa `P_penalty > 1`, com valor 2 no artigo.
- O sinal negativo da atualização após sobrepredição ajuda a tornar o modelo mais conservador.
- Registrar exatamente a condição *predicted-safe*, o KPI realizado, o limiar e o coeficiente adicionado.
- No caso de KPI do tipo **menor é melhor**, como latência, a orientação das atualizações e do evento “falha crítica” precisa ser reformulada; **não usar a expressão de throughput sem adaptação**.

**Testes unitários:** a atualização deve ser positiva na subpredição, negativa na sobrepredição e ter módulo ampliado sob falha crítica; o número de vetores nunca pode exceder `B`.

### 3.4 Seleção de ações e dois modos [ARTIGO, Seções IV-B e IV-C]

**Modo confiante** (`A_safe` não vazio):

```text
A_safe_i = { a ∈ {0..C} : f_τ_i(z_i, a) ≥ Λ_i }
score_i(a) = f_τ_i(z_i, a) − γ·(a/C)
a_base_i = argmax_(a em A_safe_i) score_i(a)
```

> Atenção: a fórmula publicada **maximiza o KPI previsto penalizado pelo custo**, e não simplesmente escolhe o menor `a` pertencente ao conjunto seguro. **Implementar a fórmula antes de testar qualquer variante “mínimo PRB seguro”.**

**Modo incerto** (`A_safe` vazio):

- Com probabilidade `ε=0,1`, amostrar ação de uma política Softmax baseada no KPI previsto e na temperatura `T_softmax`.
- Com probabilidade `1−ε`, buscar os `K=10` estados passados mais próximos e escolher ação conhecida associada a resultado satisfatório, com a regra de escore apresentada na Seção IV-B.
- Logar **qual** caminho foi acionado, inclusive quando memória vazia, sem vizinhos ou sem ação anteriormente bem-sucedida. Tratar essas bordas com regras declaradas como hipóteses de implementação.

**Margem de segurança**:

```text
a_final_i = a_base_i + round(σ_i · M_i)
```

O quantil `τ_i` e a margem `M_i` são adaptados conforme violações recentes: mais violações → **τ diminui** e **M aumenta**; quase nenhuma violação → ajuste conservador em sentido inverso. O PDF **não detalha todos os ganhos e janelas**: implementar uma função parametrizada, submetida a testes e identificada como reconstrução.

**Coordenação original**:

1. Todos os agentes propõem ações finais.
2. Se `Σ_i a_i > C`, aplicar **projeção proporcional** para retornar ao conjunto viável de capacidade.
3. Resolver arredondamento inteiro sem permitir soma superior a `C`.
4. Registrar ação proposta, ação projetada, déficit introduzido por slice e necessidade de projeção.
5. Aplicar *virtual update* negativo à proposta reduzida, conforme a lógica da Seção IV-B; o peso exato é **não documentado**.
6. Executar a ação **projetada**, observar o resultado e atualizar os regressores.

**Observação científica:** o artigo garante uma restrição sobre soma de PRBs após projeção; isso **não demonstra automaticamente** que cada ação projetada continuará satisfazendo o limiar previsto no conjunto seguro. Essa será uma hipótese concreta a testar.

### 3.5 Pseudocódigo de referência para reprodução

```text
PARA cada episódio, resetar ambiente e controladores segundo protocolo:
  inicializar memórias SQR_i, normalizadores, τ_i, M_i
  PARA t = 1..T:
    observar estado completo (sem conhecer resultado futuro)
    PARA cada slice i:
      construir z_i e candidatas a∈{0..C}
      prever f_i(z_i,a) e σ_i(z_i,a)
      construir A_safe_i
      se A_safe_i não vazio:
        a_base_i ← argmax [f_i(z_i,a) − γ*a/C]
      senão:
        a_base_i ← Softmax ou Safety-First KNN
      a_request_i ← clip(a_base_i + round(σ_i*M_i), 0, C)
    SE soma(a_request)>C:
      a_exec ← projeção_proporcional_inteira(a_request, C)
    SENÃO:
      a_exec ← a_request
    ambiente executa a_exec por 100 ms de tempo simulado
    observar KPIs e violations medidos, inclusive por UE
    PARA cada slice i:
      adicionar novo vetor de suporte com atualização de pinball loss
      reforçar sobrepredição em falhas críticas
      se proposta foi reduzida: registrar e aplicar atualização virtual
      descartar pontos conforme orçamento de 2000
      ajustar τ_i e M_i com base na janela recente
    gravar logs passo/UE/slice/método/seed
```

**Ambiguidade a registrar:** o artigo combina a expressão `σ(z,a)` com a notação `σ_t` na margem, sem fixar precisamente quando a incerteza é calculada. Propõe-se avaliá-la na ação `a_base`, registrar essa escolha e verificar sensibilidade. Quando o autor esclarecer, migrar à especificação original.

### 3.6 Testes unitários obrigatórios do SQR

- Kernel Matérn: simetria, finitude, `k(x,x)=1` quando normalizado e diminuição de similaridade com distância (para parâmetros positivos apropriados).
- Quantil: em dados sintéticos com distribuição conhecida, erro de cobertura próximo ao `τ` esperado **sob hipóteses estacionárias e amostras suficientes**.
- Penalização crítica: maior atualização negativa se erro de previsão causou violação do SLA.
- Variantes: alternar `γ` deve alterar **apenas** o termo de custo na seleção de ação, sem desligar `P_penalty`.
- Orçamento: `len(memory) ≤ 2000`; *pruning* determinístico sob semente fixa.
- Projeção: `Σa_i ≤ C`, `a_i` inteiro, `a_i ≥ 0` e não maior que solicitação após projeção.
- Estado vazio: sem crashes quando não há UEs, vizinhos, ações seguras ou memória.
- Sem vazamento: decisões no passo `t` não acessam medidas realizadas somente depois do passo `t`.
- Sementes: mesmos dados de entrada e mesmo método produzem logs iguais, dentro da tolerância computacional estabelecida.

---

## 4. Organização proposta do repositório

```text
sqr-oran-study/
├── README.md
├── CITATION.cff
├── LICENSE
├── pyproject.toml                 # dependências Python versionadas
├── configs/
│   ├── paper_reference.yaml       # publicado + pendências sem preenchimento arbitrário
│   ├── reproduction_assumptions.yaml
│   ├── scenarios.yaml
│   ├── slices_embb.yaml
│   ├── slices_heterogeneous.yaml
│   └── experiments.yaml
├── docs/
│   ├── assumptions_register.md    # decisões adotadas, justificativas e impacto
│   ├── source_traceability.md     # parâmetro → seção/tabela do PDF
│   ├── architecture.md
│   ├── reproduction_report.md
│   └── experiment_protocol.md
├── src/
│   └── oran_sqr/
│       ├── agents/
│       │   ├── sqr.py              # modelo Matérn, coeficientes, previsão de quantis
│       │   ├── sqr_controller.py   # seleção, exploração, margens
│       │   ├── kbrl_adapter.py
│       │   └── simple_baselines.py
│       ├── coordination/
│       │   ├── proportional.py     # reprodução original
│       │   └── risk_aware.py       # proposta SQR-JC
│       ├── environments/
│       │   ├── python_network.py
│       │   ├── traffic.py
│       │   ├── channel.py
│       │   └── ns3_bridge.py
│       ├── scenarios/
│       │   ├── load_profiles.py
│       │   └── slice_specs.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── statistics.py
│       │   └── plots.py
│       └── infrastructure/
│           ├── seeds.py
│           ├── logging.py
│           └── config.py
├── ns3/
│   ├── README.md                   # checkout ns-3 + NR e integração
│   ├── scratch/sqr-oran.cc         # cenário C++ com decisões periódicas
│   ├── bridge/                     # comunicação com Python
│   └── traces/                     # artefatos pequenos de validação
├── scripts/
│   ├── run_reproduction.py
│   ├── calibrate_load.py
│   ├── run_experiments.py
│   ├── analyse.py
│   └── reproduce_figures.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── parity/
│   └── fixtures/
├── results/                       # dados brutos versionados por manifesto, não por Git
└── manifests/                     # hash de código, config, seeds e dependências
```

### Dependências Python sugeridas [PROPOSTA]

- `numpy`, `scipy`: vetores, distâncias, kernel Matérn, otimização/testes.
- `pandas` ou `polars`, `pyarrow`: logs de experimentos/Parquet.
- `matplotlib`: gráficos das figuras; `seaborn` opcional no repositório do pesquisador.
- `pyyaml`/`pydantic`: leitura e validação de cenários.
- `pytest`: testes unitários e de integração.
- `scikit-learn`: apenas ferramentas auxiliares (normalização/vizinhos/testes); **não usar estimador padrão como substituto silencioso do SQR**.
- `scipy.optimize` ou `ortools`: coordenador de otimização, caso necessário.
- `pyzmq` e `protobuf` **somente se** o adaptador ns3-gym/ZeroMQ for escolhido e compatibilizado.

**Disciplina de reprodutibilidade:** salvar versão de Python, `pip freeze`/arquivo lock, Git SHA, plataforma/CPU, opções de compilação do ns-3, Git SHA do NR, configuração e listas de sementes. Não mudar versões no meio da campanha final.

### Configuração de referência (modelo com campos pendentes)

```yaml
source: "Tuerxun & Nakao, IEEE CCNC 2026"
mode: "reimplementation_from_paper"
reference:
  n_base_stations: 1
  slices: 3
  slice_types: [eMBB, eMBB, eMBB]
  min_throughput_mbps_per_ue: [0.5, 1.0, 1.5]
  target_sla_violation_rate: 0.01
  carrier_frequency_hz: 2.0e9
  cell_radius_m: 2000
  tx_power_dbm: 30
  bs_antenna_gain_dbi: 15
  noise_db_as_printed: 9
  interference_plus_noise_dbm_as_printed: -110
  propagation_label: "urban macrocell (paper reference)"
  fading_label: "Pedestrian A"
  ue_arrival_rate_per_minute: 2
  mean_session_duration_s: 60
  control_period_ms: 100
  kernel: matern
  kernel_length_scale: null        # NÃO INFORMADO
  kernel_nu: null                  # NÃO INFORMADO
  budget_support_vectors: 2000
  tau_min: 0.01
  tau_max: 0.05
  learning_rate: 0.1
  critical_failure_penalty: 2
  epsilon: 0.1
  nearest_neighbors: 10
  cost_gamma_variants: [0, 1]
  prbs_total_C: null               # NÃO INFORMADO
  bandwidth_hz: null               # NÃO INFORMADO
  nr_numerology: null              # NÃO INFORMADO
  n_steps: null                    # NÃO INFORMADO DE FORMA INEQUÍVOCA
  initial_ues: null                # NÃO INFORMADO
  n_seeds: null                    # NÃO INFORMADO DE FORMA INEQUÍVOCA
```

Usar `configs/reproduction_assumptions.yaml` para **sobrescrever** campos nulos com decisões aprovadas, e **não** editar o arquivo `paper_reference.yaml` para inventar valores publicados. O carregador de configuração deve falhar explicitamente caso ainda existam campos essenciais nulos ao iniciar uma execução.

---

## 5. Fase 1 — Reprodução em Python do cenário original

### 5.1 Entrega-alvo

Produzir os resultados de **KBRL × SQR (`γ=0`) × SQR (`γ=1`)** num mesmo ambiente Python calibrado para a referência de **três eMBB**.

**Ordem prática:**

1. Criar `paper_reference.yaml`, registrar pendências e definir controles de consistência.
2. Implementar ambiente temporal: chegadas Poisson, tempos de sessão exponenciais, usuários por slice, canal e fila/tráfego coerentes.
3. Instrumentar throughput por UE, throughput da slice, mínimo/percentil 5, PRBs ocupados e violações.
4. Implementar seleção de ação/atualização SQR seguindo a Seção IV; validar com dados sintéticos antes do ambiente completo.
5. Adicionar ambas variantes de `γ` usando a **mesma** implementação do regressor e as mesmas condições de execução.
6. Integrar KBRL mantendo **a mesma definição de ação, ambiente e observações comparáveis**.
7. Executar pequenos *smoke tests*, perfilar tempo por decisão, corrigir erros e só então rodar sementes finais.

### 5.2 Referência para KBRL

O repositório dos autores do KBRL está publicado em:  
[https://github.com/jjalcaraz-upct/network-slicing](https://github.com/jjalcaraz-upct/network-slicing)

O README documenta ambiente estilo Gym, `kbrl_control.py`, `algorithms/kernel.py`, `algorithms/projectron.py`, `channel_models.py`, `traffic_generators.py` e scripts como `experiments_kbrl.py`. **Reutilizar/adaptar a implementação KBRL**, mas **não importar os resultados do repositório como se viessem automaticamente do cenário CCNC 2026**: os cenários, estados, controle e granularidade precisam ser harmonizados. O repositório tem dependências historicamente antigas; encapsular ambiente de reprodução ou portar com testes, registrando modificações.

### 5.3 Controle de justiça experimental

Para cada semente, todos os métodos devem receber a **mesma sequência exógena** de chegadas de UEs, mobilidade, geração de pacotes e canal (na medida suportada pela implementação), com caminhos de randomização **separados do agente**. Assim, explorar mais/menos ações não muda artificialmente a dificuldade do cenário.

Não usar resultados medidos de um método para calibrar somente outro. Hiperparâmetros devem ser escolhidos em sementes/cenários de desenvolvimento, **nunca** diretamente no conjunto final de teste.

### 5.4 Critério de conclusão da reprodução

- [ ] Figuras conceitualmente equivalentes à **Fig. 2**, **Fig. 3** e **Fig. 4**: evolução da taxa de violação, violação acumulada e distribuição de PRBs.
- [ ] Resultados das três abordagens no **mesmo** ambiente e com métricas comuns.
- [ ] Todas as decisões `[PENDENTE]` resolvidas por fonte ou registradas em `assumptions_register.md`.
- [ ] Relatório de diferenças entre o ambiente próprio e a descrição do PDF.
- [ ] Testes automáticos passando e logs de tempo por passo/latência computacional.
- [ ] Se tendências não coincidirem, executar diagnóstico de sensibilidade antes de afirmar que “o artigo está errado”.

**Não exigir** exatamente a porcentagem de economia ou violação descrita no texto, pois vários parâmetros do estudo original não foram publicados.

---

## 6. Fase 2 — Quatro regimes de carga e mudanças de contexto

### 6.1 Definição quantitativa [PROPOSTA]

Evitar rótulos vagos como “normal” sem métrica de carga. Definir uma variável operacional de referência:

```text
ρ_t = demanda_estimativa_de_PRBs_no_período / C
```

A **demanda estimada** deve ser calculada por uma regra independente do algoritmo avaliado (por exemplo, tráfego oferecido e MCS/canal de referência), para que a própria política não altere o rótulo do cenário. Alternativamente, usar tráfego oferecido em Mbps em relação a uma capacidade de referência fixa e documentada.

Faixas **iniciais apenas para calibração** (não são do artigo):

| Cenário | Ocupação/demanda de referência sugerida | Expectativa |
|---|---:|---|
| Baixa | `ρ < 0,35` | Há capacidade ociosa |
| Normal | `0,35 ≤ ρ < 0,70` | Carga atendível |
| Estressada | `0,70 ≤ ρ ≤ 1,00` | Pouca margem para erro |
| Congestionada | `ρ > 1,00` | Demandas podem ser inviáveis simultaneamente |

**Calibrar essas faixas com um piloto**, variando chegada de UEs, taxa de bits oferecida e tráfego por classe. **Aumentar apenas a taxa Poisson sem checar a capacidade real pode não gerar congestionamento.** Não adulterar taxa de chegada ou PRBs entre algoritmos da mesma comparação.

### 6.2 Experimentos de carga

- **E2-A Estacionário:** quatro execuções independentes, cada uma com carga baixa, normal, estressada e congestionada; manter três eMBB e SLA original.
- **E2-B Degrau:** baixa → normal → estressada → congestionada → normal; cortes com tempo predefinido, **sem resetar o controlador** ao mudar a carga.
- **E2-C Rajadas:** chegada de UEs ou tráfego com picos temporários controlados, para verificar estabilidade e velocidade de recuperação.
- **E2-D Mudança de distribuição:** alterar perfis de mobilidade/canal mantendo carga média semelhante, se o cronograma permitir; distinguir mudança de tráfego de mudança de canal.

### 6.3 Variáveis de resposta específicas de transições

- Violações na janela anterior e posterior ao evento de mudança.
- **Violações acumuladas adicionais** até recuperar o SLA.
- **Tempo de recuperação:** primeira janela de duração definida (ex.: `W` períodos) com violação abaixo de `δ` e estabilidade subsequente. Definir `W` antes de rodar.
- Fração de decisões com modo incerto, tamanho de margem `M`, distribuição de `τ`, eventos de projeção e magnitude das reduções por slice.
- Regret/custo de adaptação só se houver referência definida e computável; não introduzir métrica sem semântica.

### 6.4 Resultado científico esperado da etapa

Estabelecer **empiricamente** se o SQR original já é robusto ou se possui falhas observáveis, **onde** e **por quê**. Não assumir que o método necessariamente piorará em congestionamento; isso é uma hipótese a testar.

---

## 7. Fase 3 — Adicionar eMBB, URLLC e mMTC sem inflar o escopo de QoS

### 7.1 Ordem de integração

1. **H1 (núcleo viável):** uma slice eMBB com SLA de **throughput** e uma URLLC com SLA de **latência**.
2. **H2 (generalização completa, se os modelos forem validados):** adicionar mMTC com **taxa de sucesso de pacotes/conexões** ou indicador explícito de acesso. Não inventar um SLA de throughput por UE para mMTC apenas para reutilizar código.
3. **H3 (opcional):** cenários mistos com várias slices do mesmo tipo e sensibilidades de prioridade.

**Nota:** throughput e latência **são métricas de QoS**. A decisão de escopo é focar em **um KPI principal por classe**, sem alegar avaliação exaustiva de cinco ou mais KPIs.

### 7.2 Segurança com direções de KPI opostas

O SQR original usa um KPI em que **maior é melhor**, com restrição:

```text
Q_tau(T_i | z,a) ≥ throughput_mínimo_i
```

Para **URLLC**, se `D_i` é atraso em ms, menor é melhor. Uma extensão natural é usar:

```text
Q_(1−δ_i)(D_i | z,a) ≤ latência_máxima_i
```

Para **mMTC**, se `S_i` é taxa de sucesso (0..1), maior é melhor:

```text
Q_(δ_i)(S_i | z,a) ≥ sucesso_mínimo_i
```

Essas são **formulações propostas**, não a implementação dos autores. Definir quantil, unidade, nível de agregação e SLA quantitativo **antes** da experimentação. **O alvo genérico de 1% de violações do experimento eMBB não é, por si só, uma meta de confiabilidade URLLC:** os limiares e orçamentos de risco devem ser justificados para o serviço estudado. O KPI de URLLC pode ser atraso por pacote, percentil elevado da latência ou probabilidade de cumprir deadline; não confundir diretamente um quantil de atraso com uma garantia formal fim a fim.

**Implementação pragmática:** criar uma classe `SlaSpec` com `name`, `direction` (`higher_is_better`/`lower_is_better`), `threshold`, `quantile_level`, `violation_budget`, `aggregation_scope`. Assim o mesmo módulo de previsão pode usar cauda inferior ou superior sem inverter sinais incorretamente.

### 7.3 Casos de teste para a generalização

- Saturação eMBB não pode ser avaliada apenas pela latência URLLC.
- Latência URLLC deve medir filas e entrega de pacotes no tempo, não ser inferida apenas pelo número de PRBs.
- mMTC precisa incluir **densidade de dispositivos**, chegada/eventos, sucesso de acesso e gargalos relevantes; não afirmar fidelidade de mMTC se só houver um fluxo TCP genérico.
- Registrar violação **por UE/pacote** e por slice, e explicitar agregação temporal.
- Prioridade de URLLC deve vir de uma política científica definida, não de codificação oculta em peso arbitrário.
- Se `Σ demandas mínimas > C`, registrar **inviabilidade** e medir severidade/justiça, em vez de exigir SLA impossível.

---

## 8. Fase 4 — Nossa proposta: coordenação conjunta orientada por risco (`SQR-JC`)

### 8.1 Lacuna operacional a investigar

No algoritmo original, cada agente estima um conjunto próprio de ações seguras `A_safe_i`. Depois os vetores são coordenados por **redução proporcional** caso `Σa_i > C`. Uma ação individual prevista segura **pode** tornar-se insegura após a redução; o artigo introduz uma penalização virtual para ensinar o agente, mas não demonstra que a projeção preserve imediatamente todas as restrições probabilísticas.

**Pergunta:** uma coordenação global que considere explicitamente o risco de SLA e a viabilidade pode reduzir a degradação em congestionamento, especialmente com SLAs heterogêneos?

### 8.2 Princípio de arquitetura [PROPOSTA]

**Preservar o regressor SQR por slice** e substituir o tratamento final do conflito de capacidade:

```text
ESTADO + KPIs
      ↓
SQR por slice: quantis + incerteza por ação
      ↓
Conjuntos candidatos (ações/risco previsto)
      ↓
COORDENADOR INTER-SLICE SQR-JC
  - capacidade total C
  - viabilidade das restrições por classe
  - risco ou déficit de SLA
  - penalização por consumo de PRBs
  - prioridades explícitas e justificadas
      ↓
Vetor conjunto de PRBs viável
      ↓
Simulador e feedback dos UEs
      ↓
Atualização online dos modelos
```

**Contribuição separável:** o objetivo é demonstrar ganhos causados pelo **coordenador**, não por mudar simultaneamente regressor, canal, SLA e recompensa.

### 8.3 Duas modalidades de coordenação, em ordem de complexidade

**Modo A — Viabilidade prevista** (quando é possível satisfazer os requisitos):

1. Para cada slice, computar ações candidatas e o respectivo quantil de risco.
2. Verificar se existe combinação com `Σa_i≤C` e com todas as restrições previstas satisfeitas.
3. Se existir, escolher combinação de menor consumo (ou compromisso consumo/folga), usando programação dinâmica/inteira ou busca exata para `L` e `C` pequenos.
4. Se possível, considerar robustez via margem baseada na incerteza; não chamar isso de garantia probabilística calibrada sem calibração.

**Modo B — Inviabilidade prevista** (não existe combinação segura):

1. Permitir déficits controlados e calculá-los na **direção correta do KPI**.
2. Minimizar uma função explícita que combine **severidade normalizada das violações previstas, justiça/prioridades e consumo**.
3. Registrar quais slices foram sacrificadas e quanto; proibir promessas de SLA que a capacidade não suporta.
4. Reavaliar o resultado observado após a alocação e atualizar os modelos.

### 8.4 Formulação sugerida [PROPOSTA, NÃO consta do artigo]

Para cada slice `i`, estimar o déficit normalizado `d_i(a_i)≥0`:

```text
se maior é melhor: d_i(a_i) = max(0, (Λ_i − Q_i(a_i)) / escala_i)
se menor é melhor: d_i(a_i) = max(0, (Q_i(a_i) − Λ_i) / escala_i)
```

Resolver, em espaço discreto de candidatos:

```text
min_{a_1,...,a_L} Σ_i [w_i · d_i(a_i)] + λ·(Σ_i a_i / C)
sujeito a: 0 ≤ a_i ≤ C; a_i inteiro; Σ_i a_i ≤ C
```

- `w_i`: peso de criticidade **pré-declarado**; testar sensibilidade para evitar injustiça arbitrária.
- `λ`: compromisso entre risco e recursos; calibrar em desenvolvimento e fixar no teste.
- `escala_i`: normalização para não comparar diretamente Mbps com milissegundos.
- Se o sistema for previsto como viável, usar prioritariamente as **restrições duras previstas** e minimizar consumo; recorrer aos déficits apenas quando não houver solução viável prevista.
- Não é obrigatório usar DRL, PPO ou DDQN.

**Limitação essencial:** `Q_i(z_i,a_i)` foi estimado a partir de estado local e resumo global. Se o desempenho de uma slice depende fortemente das ações das outras, a soma separável dos déficits pode ser imprecisa. Registrar isso e testar calibração; se necessário, adicionar ao estado informação da **alocação conjunta executada anteriormente** ou usar reavaliação iterativa após coordenação, mantendo comparação controlada. Não anunciar garantias de segurança só com esse problema simplificado.

### 8.5 Implementação incremental do coordenador

- **JC-0:** projeção proporcional original (controle).
- **JC-1:** escolher combinação por viabilidade prevista + mínimo de PRBs, sem prioridade diferenciada.
- **JC-2:** coordenar com déficits normalizados quando o conjunto previsto seguro for vazio.
- **JC-3:** acrescentar prioridade de classe e prevenção de fome (*starvation*) verificável.
- **JC-4 (opcional):** penalização de reconfiguração excessiva `||a_t−a_(t−1)||`, somente se houver resultados que justifiquem.

A vantagem de separar JC-1/2/3 é permitir **ablação** e apontar qual componente realmente melhora os resultados.

### 8.6 Hipóteses testáveis, formuladas antes dos experimentos

- **H1 (coordenação):** sob competição por PRBs, JC-1/JC-2 diminui violações induzidas pela projeção em comparação com o SQR original.
- **H2 (heterogeneidade):** em eMBB+URLLC, o método proposto reduz a severidade/tempo de recuperação das violações sob transições de carga.
- **H3 (eficiência):** os ganhos de segurança não exigem consumo excessivo de PRBs, quando comparados em **níveis de atendimento de SLA equivalentes**.
- **H4 (execução):** o tempo de decisão adicional permanece compatível com o período de controle considerado, dentro da plataforma experimental escolhida.

**Hipóteses não são resultados garantidos**. Se a projeção original já funcionar melhor sob todos os regimes, reportar isso e rever a necessidade/novidade da proposta.

---

## 9. Fase 5 — ns-3 com 5G-LENA: plano de simulação, não apenas figura arquitetural

### 9.1 Por que usar ns-3/5G-LENA

A implementação em Python permite depuração, ablação rápida e comparação algorítmica. A validação em **ns-3 com NR/5G-LENA** permite estudar agendamento realista de transmissão, canal e mobilidade, geração de pacotes e latência observada. **Python puro e 5G-LENA não são automaticamente ambientes numericamente equivalentes**; os resultados devem ser comparados como validação de transferência, não sobrepostos sem esclarecer modelos e granularidade.

### 9.2 Versões: duas pilhas possíveis — escolher UMA e congelar [PESQUISA EXTERNA]

**Pilha de simulação NR pura (sugestão para começar):** o guia oficial 5G-LENA informa compatibilidade **5G-LENA v5.1 / ns-3.48** e **v4.2 / ns-3.47**. Verificar tags exatas e notas de versão antes do clone, manter o par compatível e registrar SHA.  
Documentação oficial: <https://cttc-lena.gitlab.io/nr/html/>  
Fonte do módulo: <https://gitlab.com/cttc-lena/nr>

**Pilha com Near-RT RIC e interfaces E2 (fase avançada):** o projeto <https://github.com/Orange-OpenSource/ns-O-RAN-flexric> documenta integração FlexRIC/E2 com um *fork* NR LENA **v3.3**, além de versões específicas de E2AP e E2SM. **Não assumir** compatibilidade automática entre o NR v5.1 e esse fork anterior. Se essa fase for necessária, seguir os commits/instruções **da pilha integrada como conjunto**.

A implementação de `ns3-gym` pode facilitar troca de observações e ações Python ↔ ns-3, mas exige verificação da versão: <https://github.com/tkn-tub/ns3-gym>. Não instalar cegamente um tutorial antigo sobre uma versão mais recente sem teste de compatibilidade.

### 9.3 Etapas de preparação do ns-3 [CHECKLIST]

- [ ] Escolher e documentar sistema operacional, compilador, CMake, Python, versão de ns-3 e versão 5G-LENA **compatíveis**.
- [ ] Instalar dependências **segundo README da versão selecionada**, compilar módulo NR e executar exemplos básicos.
- [ ] Executar teste de uma gNB, um UE, tráfego conhecido e coleta de throughput/latência para conferir unidades.
- [ ] Expandir para múltiplos UEs e três classes de tráfego, mantendo **uma gNB** na primeira validação.
- [ ] Congelar parâmetros rádio com rastreabilidade (frequência, bandwidth, numerologia, modelo de canal, potência, antenas, schedulers, RLC, filas).
- [ ] Instrumentar o número de PRBs efetivamente programados por slice no MAC e a ocupação temporal.
- [ ] Implementar um mecanismo de alocação por slice **executável** pelo scheduler; testar se a soma de PRBs nunca excede capacidade.
- [ ] Integrar o controlador SQR por interface controlada e medir overhead de troca e tempo de decisão.
- [ ] Validar dados com um baseline simples antes dos experimentos completos.

### 9.4 A questão crítica do scheduler / PRBs

No experimento, `a_i` significa um **orçamento de PRBs por slice em cada período de controle**. Isso exige um mecanismo operacional no simulador.

Possíveis desenhos:

1. **Scheduler customizado com cotas por slice:** estender o scheduler NR/MAC para receber quotas atualizadas a cada decisão e restringir os grants em PRBs aos UEs pertencentes à slice. É o caminho mais fiel conceitualmente, mas requer C++ e testes de scheduler.
2. **Agendamento por conjuntos de recursos/BWPs:** pode simplificar a separação, porém uma BWP fixa **não é equivalente automaticamente** a realocar dinamicamente um orçamento de PRBs a cada 100 ms. Documentar se usar essa aproximação.
3. **Controlador fora de banda:** Python gera cotas que um adaptador C++ aplica no scheduler durante a simulação. É necessário demonstrar que o simulador realmente respeitou as cotas e que foram coletados PRBs **utilizados**, não só pedidos.

**Nunca declarar “alocação dinâmica de PRBs implementada” somente porque alterou prioridade, taxa de tráfego ou número de UEs.** Validar no nível MAC a aplicação da decisão.

### 9.5 Loop de controle Python ↔ ns-3

```text
ns-3 (C++): executa eventos por Δt = 100 ms simulados
          ↓ observação da janela anterior (timestamp, KPIs, UEs, filas)
Python:   forma estado e calcula SQR / SQR-JC
          ↓ vetor de cotas PRB válido (timestamp, C, slice_id→PRBs)
ns-3:     atualiza scheduler/cotas ANTES da próxima janela
          ↓ PHY/MAC/RLC/PDCP; UEs recebem pacotes
ns-3:     entrega resultados realizados da janela (throughput, atraso, PRBs)
Python:   atualiza regressores e logs
```

- Garantir **sincronização causal** e que não haja consumo de KPI futuro.
- Separar **tempo simulado** (100 ms) de **tempo real de inferência/IPC**, registrando ambos.
- Prevenir deadlock/timeouts e tratar mensagens inválidas; se Python falhar, política de fallback determinística e logada.
- Determinar se o controlador age a cada 100 ms **simulados** ou a cada 100 slots NR; para NR SCS/numerologia diferentes, esses números não são sinônimos.

### 9.6 Instrumentação necessária no ns-3

**Por UE:** `ue_id`, `slice_id`, gNB, posição, chegada/saída, bytes TX/RX, throughput de janela, atraso de pacotes, drops, CQI/SINR/SNR quando disponível.  
**Por slice:** UEs ativos, demanda oferecida, backlog/filas, taxa de violação, percentis de throughput, quantis de atraso, utilização de PRBs, tipos/prioridades de SLA.  
**Por ação:** propostas locais, projeção original ou alocação conjunta, PRBs agendados reais, eventos de insuficiência e duração/latência do controlador.

**Exemplo de logs em uma janela:**

```text
run_id, seed, scenario, step, sim_time_ms, controller,
slice_id, slice_type, ue_count,
prb_requested, prb_executed, prb_scheduled,
throughput_mbps_p05, throughput_mbps_min,
latency_ms_p99, packet_success_rate,
sla_threshold, sla_violation_flag, violation_severity,
tau, margin_M, novelty_sigma, predicted_quantile,
projection_triggered, inference_wall_ms
```

Salvar dados em CSV para depuração e Parquet para volume maior; para resultados per-packet/per-UE, usar tabelas separadas ligadas por `run_id`, `step`, `ue_id`.

### 9.7 O que significa “validar O-RAN”

Três níveis diferentes, que não devem ser confundidos:

- **Alinhamento arquitetural:** a lógica seria uma xApp no Near-RT RIC, com controle em escala próxima a 100 ms. Isso é **desenho**, não integração.
- **Integração funcional:** dados chegam por uma interface E2/KPM e decisões são efetivamente aplicadas via controle suportado pelo simulador/fork; requer testes E2AP/E2SM-RC ou mecanismo equivalente explicitado.
- **Implantação real/testbed:** FlexRIC/testbed com rádio ou HIL e medições práticas; **fora do escopo obrigatório** do primeiro artigo, salvo nova decisão.

A referência SQR descreve o primeiro nível e relata simulação. A validação real em Colosseum/Arena é trabalho futuro declarado pelos autores.

---

## 10. Matriz experimental consolidada

| ID | Ambiente | Slices | Tráfego/carga | Métodos | Pergunta respondida |
|---|---|---|---|---|---|
| **E0** | Python sintético | Distribuições controladas | Estacionário | SQR isolado | O regressor e as regras operam corretamente? |
| **E1** | Python referência | 3 eMBB (0,5/1/1,5 Mbps/UE) | Similar ao artigo | KBRL; SQR `γ=0`; SQR `γ=1` | Reproduzimos tendências das Figs. 2–4? |
| **E2** | Python referência | 3 eMBB | 4 cargas estacionárias | Três métodos originais | Onde o SQR perde eficiência/segurança? |
| **E3** | Python referência | 3 eMBB | Transições + rajadas | SQR variantes + KBRL | Como o método se recupera de mudanças? |
| **E4** | Python heterogêneo | eMBB + URLLC | 4 cargas e transições | SQR adaptado + baselines | Como estender as restrições de KPI? |
| **E5** | Python heterogêneo | eMBB + URLLC + mMTC | 4 cargas selecionadas | SQR adaptado + baselines | Generaliza aos três tipos de slice? |
| **E6** | Python heterogêneo | eMBB/URLLC/(mMTC) | E2–E5 | SQR original/projeção; **SQR-JC** | A coordenação reduz violações? |
| **E7** | Python heterogêneo | Mesmo de E6 | Mesmo de E6 | JC-0/1/2/3 e variantes | Qual componente gera o ganho? |
| **E8** | ns-3 + 5G-LENA | 3 eMBB, depois eMBB+URLLC | Subconjunto representativo de E1/E6 | SQR; SQR-JC; baseline | O resultado se sustenta sob PHY/MAC detalhado? |
| **E9** *(opcional)* | ns-3 + Near-RT RIC | Subconjunto | Carga representativa | xApp SQR-JC | Há viabilidade de integração O-RAN? |

**Dependências lógicas:** `E0 → E1 → E2/E3 → E4 → E6/E7 → E8`. E5 deve acontecer **antes** de declarar resultados sobre os três tipos de slices. E9 é trabalho de validação adicional, não pré-requisito da contribuição algorítmica.

### Baselines e cuidados de equivalência

- **Obrigatórios:** SQR `γ=0`, SQR `γ=1`, `SQR-JC`, e pelo menos um baseline externo validado (KBRL é natural porque o artigo o utiliza).
- **Desejáveis:** RR/PF como escalonamento convencional **com definição explícita de como suas decisões geram divisão inter-slice**; RSLAQ somente se arquitetura/ações, restrições e capacidade forem comparáveis.
- **Não assumir** que RR/PF/BCQI são políticas completas de slicing; eles podem ser políticas de escalonamento **intra-slice** ou de UE, e exigem adaptação legítima ao nível de controle em análise.
- **Comparação justa:** mesmas trajetórias/canal, orçamento total `C`, UEs, métricas, granularidade e configuração do scheduler; diferença limitada ao algoritmo de decisão.

---

## 11. Protocolo estatístico e métricas

### 11.1 Métricas principais

| Métrica | Definição operacional | Granularidade |
|---|---|---|
| Taxa de violações de SLA | `N_violações / N_unidades_elegíveis` | Por UE/pacote ou por slice-período — **denominador explícito** |
| Violações acumuladas | Soma de indicadores até `t` | Curva por execução e agregado |
| Severidade de violação | Déficit normalizado em relação ao limiar | Por slice e por evento |
| Consumo de PRBs | Média/soma de **PRBs efetivamente programados** por janela | Por método/slice/estado de carga |
| Recursos solicitados versus utilizados | `a_request`, `a_exec`, `PRB_MAC_real` | Por decisão |
| Tempo de recuperação | Duração até janela voltar a atender taxa-limite por `W` decisões | Após transições |
| Taxa/magnitude de projeção | Frequência e tamanho das reduções impostas pelo coordenador | Por decisão e slice |
| Tempo de decisão | P50/P95/P99 do tempo real Python+otimizador e IPC separado | Por passo |
| Justiça entre slices | Índice de satisfação/violação normalizado por requisito | Por cenário |
| Eficiência | Throughput útil total dividido por PRBs-tempo consumidos, se unidade consistente | Por janela/cenário |

**Distinguir** (a) PRBs reservados por slice, (b) PRBs efetivamente transmitidos pelo scheduler e (c) capacidade total disponível. Reportar pelo menos (a) e (b); a Fig. 4 não esclarece todas essas distinções operacionais.

### 11.2 Exemplo de definições sem ambiguidade

```text
violation_UE_i,u,t = 1[throughput_i,u,t < SLA_throughput_i]
violation_slice_i,t = 1[KPI_agregado_i,t viola SLA_i]

V_UE = Σ violation_UE / #observações UE-tempo elegíveis
V_slice = Σ violation_slice / #observações slice-tempo elegíveis

CumulativeViolations(t) = Σ_(k≤t) Σ_i violation_slice_i,k
MeanPrbUsage = (1/T) Σ_t Σ_i PRB_scheduled_i,t
```

**Não comparar taxas UE-tempo e slice-tempo como se fossem numericamente idênticas.** Janelas sem UE devem ter regra de elegibilidade pré-definida, preferencialmente excluídas do denominador de QoS de UE com registro em separado.

### 11.3 Sementes e significância

- **Desenvolvimento:** começar com 3–5 sementes para depuração e perfis de tempo.
- **Experimento principal:** sugerir **≥10 sementes independentes** e, se orçamento permitir, **20 ou mais** nas configurações finais. São escolhas **propostas**, não do artigo.
- Aplicar sementes **pareadas** entre algoritmos por cenário; relatar média/mediana e **intervalo de confiança 95%** obtido por reamostragem na **unidade de execução/seed**, não por passos temporais autocorrelacionados tratados como réplicas independentes.
- Explicitar variação, não divulgar apenas o melhor seed.
- Não escolher método vencedor por uma diferença pequena sem CI/análise de tamanho de efeito.
- Registrar análise de sensibilidade a `γ`, orçamento `B`, `τ`, margem `M`, pesos de prioridade, intensidade da carga e parâmetros de kernel.

### 11.4 Figuras/tabelas a produzir

1. **Réplica Fig. 2:** taxa de violação ao longo do tempo, com referência à meta de 1%.
2. **Réplica Fig. 3:** violações acumuladas por método, com IC entre execuções.
3. **Réplica Fig. 4:** gráfico violino/boxplot de PRBs usados por decisão e método.
4. **Novo:** violações por slice × cenário (heatmap ou tabela).
5. **Novo:** violações acumuladas em torno da mudança de carga (janela relativa ao evento).
6. **Novo:** tempo de recuperação e severidade em congestionamento.
7. **Novo:** uso de PRBs versus taxa de violação (fronteira de trade-off).
8. **Novo:** impacto da coordenação (ações individualmente seguras antes × depois da projeção).
9. **Novo:** latência de inferência P95 e custo de CPU/memória.
10. **Novo:** tabela de ablações JC-0/1/2/3 e principais hiperparâmetros.

---

## 12. Validação causal e riscos metodológicos

| Risco | Consequência | Mitigação |
|---|---|---|
| Artigo não divulga `C`, bandwidth e outros detalhes | Reprodução não única | Solicitar dados aos autores; documentar hipóteses; análise de sensibilidade |
| Modelo Python simplificado não produz latência realista | Resultado sem validade externa | ns-3/5G-LENA e modelo de pacotes/fila |
| Regressor SQR inicial com memória vazia | Exploração instável | Comportamento de fallback explicitado e testado |
| Kernel sem normalização/escala | Similaridade espúria e `σ` inválido | Features normalizadas, kernel normalizado, testes |
| Regressor aprende um cenário e é testado no mesmo seed | Viés otimista | Sementes e cenários de teste separados |
| Uma fatia domina outra em congestionamento | “Melhoria” só desloca a violação | Métricas por slice, severidade e justiça |
| Capacidade fisicamente insuficiente | SLA impossível por definição | Medir sobrecarga/inviabilidade, separar regime factível e infactível |
| Projeção proporcional não preserva SLA | Violações mesmo com ações localmente seguras | Logs antes/depois da projeção e novo coordenador |
| Diferença entre KBRL per-slice e SQR per-UE | Comparação enviesada | Definições comuns de observações e violações |
| Comparação PRBs reservados vs realmente alocados | Ganhos artificiais | Instrumentar `PRB_requested`, `executed`, `scheduled` |
| Aproximar cota PRB com BWP estática | Controle não equivalente | Scheduler com cotas dinâmicas ou declarar aproximação |
| Versões ns-3/NR/FlexRIC incompatíveis | Reprovação técnica/atraso | Fixar uma pilha, build CI e smoke tests antes de customizar |
| Otimização conjunta custa mais que 100 ms | Impraticável na escala desejada | Medir p95/p99 e usar conjunto reduzido de candidatas/DP |
| Comparar variantes mudando muitos elementos | Causa do ganho indeterminada | Ablação, one-change-at-a-time no núcleo experimental |

**Checklist de investigação científica antes de alegar novidade:** revisar literatura sobre *safe RAN slicing*, *joint/constrained resource allocation*, *risk-aware coordination*, *distributional/quantile RL* e tratamento de *feasibility projection*. O fato de o SQR não testar uma variante **não prova** que ela inexiste na literatura.

---

## 13. Cronograma por marcos e critérios de aceite (sem compromisso de datas)

| Marco | Implementador entrega | Revisão pela dupla |
|---|---|---|
| **M0 — Especificação congelada** | Tabela fonte, pendências, escolhas e baseline rodando em teste mínimo | Aprovar hipóteses sem confundir publicado e assumido |
| **M1 — SQR Python validado** | Kernel, online quantile, controles, projeção, testes e logs | Conferir sinais e fórmula `γ`, `P_penalty` |
| **M2 — Reprodução E1** | Três curvas da referência, diagnóstico de diferenças | Aprovar interpretação e cobertura de SLA |
| **M3 — Cargas E2/E3** | Quatro cenários e transições, métricas por slice | Identificar limitações observadas, sem antecipar conclusões |
| **M4 — Heterogeneidade E4/E5** | eMBB/URLLC; mMTC caso no escopo acordado | Confirmar KPI, limiares, unidade e plausibilidade do tráfego |
| **M5 — Proposta SQR-JC** | Coordenador, logs de decisões, ablações E6/E7 | Verificar hipótese e vantagem versus projeção proporcional |
| **M6 — ns-3 E8** | Cenários NR, scheduler de cotas, interface Python, KPIs reais | Avaliar fidelidade e limitações de migração |
| **M7 — Resultados finais** | Scripts, dados brutos, gráficos, CI, manifests, README reprodutível | Redigir artigo com o que foi **realmente comprovado** |

### Gatilhos de decisão (*go/no-go*)

- **Após M1:** se o modelo quantílico não passar em testes sintéticos, **não** rodar 4 cenários nem ns-3.
- **Após M2:** se o baseline não reproduzir tendências, analisar parâmetros ausentes e registrar o status *reimplementation*, sem chamar de reprodução exata.
- **Após M3:** se não houver limitação aparente no SQR, procurar testes mais informativos de coordenação/conflito, **não fabricar deterioração**.
- **Após M5:** se o ganho só ocorrer por consumo exagerado de PRBs, a proposta precisa ser revista.
- **Antes de M6:** ns-3 deve ter um scheduler/controle com aplicação observável de cota PRB; caso contrário, o experimento não valida a mesma ação.

---

## 14. Primeiro conjunto de tarefas: o que começar **agora**

### Sprint técnico de preparação — entregáveis objetivos

1. [ ] Criar Git e diretórios do projeto, configurar Python/ambiente virtual, testes e formatação.
2. [ ] Copiar a Tabela I e as equações/algoritmo para `source_traceability.md` e este YAML base.
3. [ ] Abrir `assumptions_register.md` com `C`, largura de banda, especificação de tráfego, modelo de canal, `ν/ℓ` do Matérn, regra de pruning, `M`, `τ`, `T_softmax` e denominadores de SLA marcados como **pendentes**.
4. [ ] Verificar contato com autores / disponibilidade de código do SQR. Separar cópia do KBRL em ambiente de dependências próprio.
5. [ ] Implementar `SlaSpec`, `KernelQuantileRegressor`, `SqrController`, `ProportionalCoordinator`, `MetricsLogger` com testes sintéticos.
6. [ ] Criar ambiente Python mínimo com 3 eMBB; **sem URLLC e mMTC neste primeiro ciclo**.
7. [ ] Construir um único experimento SQR `γ=0` de curta duração e demonstrar logs e testes.
8. [ ] Rodar o mesmo seed com SQR `γ=1`, verificando se o termo de custo produz decisões plausíveis.
9. [ ] Integrar KBRL e gerar as três figuras preliminares antes de escala completa.
10. [ ] Em paralelo, fazer **smoke test** de ns-3 + NR puro; validar versão e captura de throughput/latência, sem ainda construir xApp.

### O que a dupla deve devolver para revisão antes da próxima fase

```text
[ ] link do repositório/commit
[ ] relatório de build (Python e ns-3, se disponível)
[ ] configurações YAML usadas e lista de pendências
[ ] saída de testes unitários
[ ] exemplo de log por decisão com PRBs propostos e executados
[ ] gráfico preliminar de violações por passo
[ ] gráfico preliminar de PRBs (γ=0 × γ=1)
[ ] lista de diferenças/ambiguidade do artigo que afetam o resultado
[ ] estimativa de tempo por passo + gargalos
```

**Mensagem operacional para o implementador:** *implemente primeiro a referência publicada, sem misturar nossa proposta; depois crie a extensão por cenários, depois heterogeneidade; o coordenador inteligente é uma variante independente da projeção original. Todas as escolhas de parâmetros não publicadas têm de aparecer em um registro de hipóteses.*

---

## 15. Perguntas a encaminhar aos autores do SQR

Solicitar, de modo objetivo:

1. Repositório/código e identificador da versão usada na CCNC 2026.
2. Simulador, versão, modelo de camada MAC/scheduler e configuração de banda/SCS/numerologia.
3. Capacidade total `C` de PRBs, largura de banda e interpretação do número de RBs na Fig. 4.
4. Features completas por slice e globais, forma de normalização, e mecanismo de simulação dos UEs.
5. Processo de tráfego Poisson: 2 UEs/min no total ou por slice; distribuição inicial de UEs; direção DL/UL e processo de pacotes.
6. Kernel Matérn: `ν`, `length_scale`, amplitude, normalização e atualização.
7. Algoritmo exato de *pruning* da memória de 2.000 vetores.
8. Valores e funções de adaptação de `M_t`, `τ_t` e `T_softmax`; como calculam `σ` na escolha da margem.
9. Condição, coeficiente e aplicação da atualização virtual após projeção.
10. Quantidade de seeds, episódios, passos, períodos de aquecimento e quantas séries compõem as figuras.
11. Definição precisa dos indicadores per-UE e per-slice nas Figs. 2–3 e modo como KBRL foi harmonizado.
12. Comparações exatas que fundamentam os percentuais de redução da conclusão.

Até obter resposta, manter as respostas como **desconhecidas** e adotar hipóteses rastreáveis.

---

## 16. Fontes e referências para implementação

### Primária — artigo objeto de reprodução

- Tuerxun, A.; Nakao, A. *Safe RAN Slicing in O-RAN: Minimizing SLA Violations via Model-Based Reinforcement Learning*. IEEE CCNC 2026. DOI: <https://doi.org/10.1109/CCNC65079.2026.11366589>. Referências operacionais internas: Seção III (estado, ação, problema), Seção IV (regressor e controle), Tabela I e Seção V (avaliação), Fig. 1 (arquitetura), Fig. 2–4 (resultados), Seção VI (limitações futuras).

### Externas — software e documentação (não são parâmetros do SQR)

- Código do KBRL e ambiente Gym associado a Alcaraz et al.: <https://github.com/jjalcaraz-upct/network-slicing>.
- Documentação oficial 5G-LENA (compatibilidade release NR/ns-3): <https://cttc-lena.gitlab.io/nr/html/>.
- Repositório oficial NR 5G-LENA: <https://gitlab.com/cttc-lena/nr>.
- ns-3: <https://www.nsnam.org/documentation/>.
- Projeto integrado ns-O-RAN-flexric (FlexRIC, E2, 5G-LENA em fork próprio): <https://github.com/Orange-OpenSource/ns-O-RAN-flexric>.
- Documentação dos KPIs do fork O-RAN/5G-LENA: <https://github.com/Orange-OpenSource/ns-O-RAN-flexric/blob/main/docs/LENA_MODEL_REPORTING_PARAMETERS.md>.
- Ponte Python ↔ ns-3 a considerar mediante compatibilidade: <https://github.com/tkn-tub/ns3-gym>.

### Validade das conclusões do novo artigo

O resultado cientificamente defensável é uma **diferença demonstrada e atribuível** à coordenação de risco em um ambiente e protocolo transparentes. A proposta **não deve afirmar** (a) reprodução exata de parâmetros omitidos; (b) implementação O-RAN/E2 apenas com uma figura de RIC; (c) segurança probabilística formal sem calibração/garantia; (d) otimização de todos os KPIs da QoS se só throughput, latência e uma métrica mMTC forem medidos.

---

## 17. Definição de pronto (*Definition of Done*) para apoiar a redação do artigo

O trabalho estará pronto para escrita de resultados quando for possível responder **com código, dados e figura**:

- [ ] O SQR foi implementado conforme as fórmulas e fluxos divulgados? Quais parâmetros foram assumidos e por quê?
- [ ] Os resultados originais foram reproduzidos qualitativamente? Se não, as divergências estão documentadas?
- [ ] O que ocorre com o SQR quando a carga muda e quando várias slices disputam PRBs?
- [ ] A projeção proporcional original faz ações previstas seguras perderem viabilidade? Com que frequência e magnitude?
- [ ] O SQR-JC reduz violações, severidade ou tempo de recuperação no **mesmo cenário e consumo comparável**?
- [ ] Qual componente da proposta causa o ganho (ablação)?
- [ ] O resultado vale para throughput e latência (e mMTC se incluída) sem sacrificar uma classe silenciosamente?
- [ ] O ganho se mantém em mais de uma semente e com incerteza estatística reportada?
- [ ] Quais resultados foram validados em Python e quais em ns-3/5G-LENA? Existe aplicação real de cotas no MAC?
- [ ] O controlador consegue decidir dentro do orçamento temporal declarado? Qual o custo de memória/CPU?
- [ ] A inovação foi comparada à literatura além do SQR e do KBRL?

**Prioridade prática resumida: `SQR Python fiel → 3 eMBB → 4 cargas e transições → eMBB+URLLC → SQR-JC → mMTC (antes de alegar três tipos de slices) → validação ns-3 → eventual integração E2`.** A ordem de mMTC versus início da integração ns-3 pode correr em paralelo; ambas exigem validação própria.
