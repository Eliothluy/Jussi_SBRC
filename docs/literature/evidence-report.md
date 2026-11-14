# Relatório de evidências — comparação de metodologias, ambientes, baselines e estatística

> **Skill:** oran-literature-review · **Artefato:** 5/6 (`evidence-report.md`) · **Data:** 2026-10-18
> **Fonte primária:** `literature-matrix.csv` (60 entradas, linhas L02–L61; toda citação neste documento usa o formato `[Lxx]` + DOI e remete à linha correspondente da matriz).
> **Contagens deste relatório foram computadas nesta sessão** por leitura direta dos campos da matriz e de `tools/search-log.jsonl` (137 entradas — 132 da sessão de busca + 5 da sessão de revisão); os comandos de agregação estão registrados na transcrição da sessão.
> **Correção (sessão de revisão, 2026-10-18):** uma auditoria externa encontrou 4 divergências de contagem/agregação — split full-text×abstract (§1), tipos de venue (§2.1), contagem de métricas de SLA/violação (§2.4) e anos de L23/L26/L27 — **todas confirmadas contra a matriz e contra o Crossref nesta sessão de revisão e corrigidas**; cada ponto corrigido está marcado **[CORRIGIDO]** com a evidência. As correções de ano/autoria alteraram também `literature-matrix.csv` (notas `CORREÇÃO` nos campos `verified` de L23, L26, L27 e L54).
> **Regras aplicadas:** claim de autor ≠ fato verificado por nós; campo ausente = "não reportado"; ausência nos 60 incluídos ≠ inexistência na literatura (ver §7).

---

## 1. Escopo e base de evidência

A matriz contém **60 entradas** resultando do funil de triagem: **104 candidatos triados → 60 incluídos → 44 excluídos** (`screening-log.csv`; 8 das exclusões por duplicata E4, razões E1–E7 e cortes de curadoria I5 registradas uma a uma).

Nível de verificação das 60 entradas (campo `verified`, classificado nesta sessão):

| Nível | n | Significado |
|---|---|---|
| Full-text lido nesta sessão | **27** **[CORRIGIDO: era 31]** | PDF/HTML integral ou preprint integral lido; campos extraídos do corpo |
| Abstract + metadados | **33** **[CORRIGIDO: era 29]** | Somente abstract (OpenAlex/Semantic Scholar/Crossref) + metadados verificados; campos não cobertos = "não reportado" |

*Classificação recomputada na sessão de revisão, linha a linha, a partir do campo `verified`. As 27 full-text: L04, L06–L08, L12–L16, L18, L20–L21, L23–L25, L27, L30–L31, L33, L37, L44–L45, L47, L55, L59–L61. Casos de fronteira, classificados como abstract+metadados: **L02** (PDF baixado, mas só pp. 1–2 — folha de rosto JSTOR + p. com abstract — lidas como imagem; campos extraídos do abstract) e **L09** (demonstração arXiv de 2 pp. da qual só a página de abstract é pública).*

**Regra de leitura**: afirmações extraídas de abstract são rotuladas como claim do autor na matriz e aqui; nenhum achado de full-text foi inferido a partir de abstract. Os 33 abstract-only concentram-se em periódicos IEEE fechados (TMC, TComm, TWC, JSAC, IEEE Access) e em venues ACM/IEEE sem OA — suas lacunas de preenchimento são **limitação de acesso**, não evidência de omissão dos autores (§7, L8).

---

## 2. Panorama quantitativo

### 2.1 Distribuição por ano e tipo de venue

| Ano | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | 1978¹ |
|---|---|---|---|---|---|---|---|---|---|
| n | 5 | 5 | 7 | 7 | 9 | 7 | 11 | 8 | 1 |

¹ Exceção seminal do critério I1 via snowballing: Koenker & Bassett, *Regression Quantiles* [L02]+DOI 10.2307/1913643 — base teórica da perda pinball usada pelo SQR [L55]+DOI 10.1109/ccnc65079.2026.11366589.

² Convenção: **ano do fascículo impresso** (`published-print` do Crossref). **[CORRIGIDO na sessão de revisão]**: L23 2022→2023 (TWC v.22 n.1, jan. 2023), L26 2023→2024 (JSAC v.42 n.2, fev. 2024) e L27 2020→2021 (JSAC v.39 n.7, jul. 2021; preprint arXiv dez. 2020) usavam o ano de registro/online-first do DOI, contra a convenção já adotada por L05 (TComm v.73 n.8, ago. 2025). Verificação: `api.crossref.org/works/<DOI>` em 2026-10-18 (registrado em `tools/search-log.jsonl`).

| Tipo de venue | n |
|---|---|
| Conferência/workshop | 31 **[CORRIGIDO: era 36]** |
| Periódico | 27 **[CORRIGIDO: era 22]** |
| Preprint/artefato (arXiv sem venue confirmado; Zenodo) | 2 ([L09]+10.48550/arxiv.2208.14877, [L50]+10.5281/zenodo.6553700) |

*Reclassificação literal dos 60 valores do campo `venue` na sessão de revisão. Critério: **periódico** = transactions/journal/magazine/open journal/access — inclui L13 (IEEE Communications Standards Magazine), L28 (OJ-COMS), L29 (TMLCN), L35/L51–L53 (IEEE Access), L45/L47 (Simulation Modelling Practice and Theory) e L54 (EURASIP JWCN); **conferência/workshop** = proceedings de conferência, workshop ou simpósio (WCNC, DySPAN, WNS3, ICTC, CoNEXT, SBrT, ICCCN, ICC Workshops, MobiCom, MILCOM, ICOIN, ICMLCN, ICNC, Globecom, MECOM, ns-3 ×7, CCNC ×2, LATINCOM, MSWiM, CIKM, ICNP ×2, BigData, EuCNC); **preprint/artefato** = L09 e L50.*

A produção de 2025–2026 (19 entradas) concentra o avanço em xApps sobre E2 real e safe/risk-aware RL — os dois eixos onde nosso artigo se posiciona.

### 2.2 Famílias de ambiente (classificação manual dos 60 valores do campo `environment`, feita nesta sessão; soma = 60)

| Família | n | Linhas |
|---|---|---|
| **ns-3 + 5G-LENA/NR (system-level 5G)** | 10 | L22, L28, L38–L43, L45, L47 |
| **Simulador próprio/numérico (não ns-3)** | 10 | L23, L24, L25, L27, L30, L32, L33, L55, L60, L61 |
| **Não reportado/plataforma não nomeada** | 17 | L05, L11, L17, L21, L26, L29, L34, L35, L36, L48, L51, L52, L53, L54, L56, L57, L58 |
| **Testbed real (OAI/srsRAN/SDR/NVIDIA)** | 8 | L03, L14, L15, L16, L19, L20, L37, L59 |
| **ns-3 + O-RAN (E2/RIC integrado)** | 4 | L06, L08, L10, L12 |
| **Colosseum/emulação SDR** | 4 | L04, L07, L09, L31 |
| **ns-3 outros módulos/ferramental** | 4 | L44, L46, L49, L50 |
| **MATLAB** | 1 | L18 |
| **n/a (teoria/survey)** | 2 | L02, L13 |

Leitura: **(i)** a família ns-3/5G-LENA (14 entradas somando as 3 variantes ns-3) é a maior base de avaliação system-level 5G da amostra — e é a família do nosso ambiente MODA; **(ii)** os 10 simuladores próprios incluem os dois polos da comparação que reproduzimos: KBRL [L23]+DOI 10.1109/twc.2022.3195570 (Python próprio, fading por traços ns-3) e SQR [L55]+10.1109/ccnc65079.2026.11366589 (simulador não publicado, "não é ns-3"); **(iii)** 17/60 (28%) nem nomeiam a plataforma — entradas abstract-only ou omissão dos autores.

### 2.3 Famílias de algoritmo (classificação manual; soma = 60)

| Família | n | Linhas |
|---|---|---|
| RL restrito/safe/risk-aware (CMDP, Lagrangiano, CVaR, interior-point) | 10 | L24–L29, L56, L57, L58, L59 |
| DRL genérico (PPO/DQN/DDPG/hierárquico) | 12 | L05, L07, L08, L15, L17, L18, L31, L33, L34, L35, L36, L37 |
| n/a (infra/plataforma/survey/teoria/calibração) | 20 | L02, L04, L06, L09, L11–L13, L16, L21, L22, L42–L47, L49, L50, L51, L54 |
| Não-RL (Lyapunov, ANN, rule-based, autonômico) | 8 | L20, L38, L39, L40, L41, L48, L52, L53 |
| Quantílico/distributional (neural: GAN-DDQN, QR-DQN, QRF) | 4 | L03, L30, L60, L61 |
| Kernel/model-based **sem rede neural** (KBRL, SQR) | 2 | L23, L55 |
| ML supervisionado (classificação de estado) | 1 | L14 |
| Não reportado | 3 | L10, L19, L32 |

Leitura central para o G8: a combinação que define o SQR — **kernel model-based + quantílico (pinball), sem rede neural** — tem exatamente **1 antecedente na amostra** (o próprio KBRL [L23]) e 1 instanciação (SQR [L55]). Os primos quantílicos neurais [L30]+10.1109/jsac.2019.2959185, [L60]+10.1109/twc.2021.3107520, [L61]+10.1109/eucnc/6gsummit63408.2025.11037178 e o QRF do mesmo grupo do SQR [L03]+10.1109/wcnc61545.2025.10978275 existem, mas sempre com rede neural e nunca acoplados a projeção de viabilidade com cota verificável no MAC.

### 2.4 Métricas e reprodutibilidade

- **17/60** entradas **[CORRIGIDO: era 14/60]** reportam métricas ligadas a SLA/violação: L05, L17, L18, L23, **L25**, L26, **L27**, L28, L30, L31, L32, **L35**, L36, L37, L52, L55, L59 (contagem por inspeção do campo `metrics`, recomputada na sessão de revisão com os termos *viola/SLA/satisfação*; a primeira contagem omitiu L25 — violações instantâneas acumuladas, L27 — probabilidade de violação de QoS — e L35 — satisfação média dos requisitos das slices). **Fronteira**: L29 (satisfação de latência/confiabilidade, sem menção explícita a SLA/violação no campo) elevaria a contagem a 18/60 sob leitura ampla de "satisfação de requisitos" — mantida fora por conservadorismo.
- **Artefatos públicos (verificação literal do campo `reproducibility` feita nesta sessão):** **12/60** citam artefato público real — L06 (3 repositórios), L12 (repo NORI + clone local conferido), L16 (repo + PR), L20 (2 repos, não acessados), **L23** (repo KBRL + hiperparâmetros conferidos no nosso port), L24 (repo citado, não inspecionado), L25 (repo citado, não inspecionado), L45 (GPLv2), L46/L50 (Zenodo; L50 com registro verificado via OpenAlex nesta sessão), L47 (5G-LENA open-source), L49 (repo SEM localizado e conferido). Casos especiais: **L07** promete "disponibilização pública futura" e a URL candidata retornou 404 nesta sessão; **L37 e L59 declaram explicitamente ausência de repositório** ("sem repositório público citado no preprint"). Os **45** restantes: "não reportado" ou acesso fechado. Destaque central: KBRL publica simulador+algoritmos+scripts ([L23]+10.1109/twc.2022.3195570, rodapé da Sec. VII) — **o baseline do SQR é reproduzível; o SQR não** ([L55]: "simulador não publicado"). NORI [L12]+10.14209/sbrt.2025.1571157244, ORANSlice [L16]+10.1145/3636534.3701544, xDevSM [L20]+10.1109/tnsm.2026.3733344 e SEM [L49]+10.1145/3345768.3355942 são as plataformas abertas da amostra; ColO-RAN [L07]+10.1109/tmc.2022.3188013 é promessa não cumprida no preprint lido (artefato da versão publicada não localizado nesta sessão).

---

## 3. Comparação de metodologias e baselines

### 3.1 Como cada família valida seus controladores

| Família | Protocolo típico | Baselines típicos | Exemplos |
|---|---|---|---|
| Simulador próprio + DRL | Treino/avaliação no mesmo ambiente fechado; métricas de recompensa/violação | Variantes do próprio método; schedulers fixos | [L24]+10.1109/jsac.2023.3336191 (3 baselines de TL); [L30]+10.1109/jsac.2019.2959185 (hard slicing, DQN); [L27]+10.1109/jsac.2020.3041405 (DDPG/TD3 c/ shaping) |
| Simulador próprio + RL restrito | Restrição explícita no treino; violação como constraint | RL sem restrição; variantes internas | [L26]+10.1109/jsac.2023.3336156 ("RL methods" sem especificação); [L57]+10.1109/icnp49622.2020.9259378; [L58]+10.1109/bigdata52589.2021.9671840 |
| Stack real/emulação + xApp | Loop fechado E2 sobre testbed/Colosseum; CDFs de latência/throughput | Allocation igual/proporcional; schedulers 3GPP; DQN | [L07]+10.1109/tmc.2022.3188013 (static + RR/WF/PF + DRL reduzido); [L15]+10.1109/iccworkshops67674.2025.11162144 (igual/proporcional/PF/DQN); [L31]+10.1109/icnc59896.2024.10556357 (DQN, Q-Learning); [L37]+10.1109/mecom67453.2025.11439664 (hard slicing, prioridade, fair, DQN) |
| ns-3/5G-LENA (sem RL) | Calibração contra curvas 3GPP/ITU; validação de scheduler | Schedulers nativos; referências 3GPP | [L47]+10.1016/j.simpat.2022.102580; [L45]+10.1016/j.simpat.2019.101933; [L39]+10.1145/3659111.3659118; [L41]+10.1145/3747204.3747214 |
| MATLAB (RSLAQ) | Cenários de carga discretos (low→stressed) | RR, PF, BCQI, método otimizado sem SLA | [L18]+10.1109/tmc.2026.3666787 |

**Observação comparativa (G8)**: nenhum trabalho da amostra que avalia controladores com SLA usa simultaneamente (a) baseline kernel model-based, (b) baseline DRL moderno e (c) baseline estático com a mesma ação no mesmo ambiente. O KBRL [L23] compara-se a 7 baselines DRL (DQN, TRPO, PPO×2, TD3, SAC, A2C, NAF) — mas no simulador Python próprio; o SQR [L55] compara-se apenas a KBRL e a variantes internas γ=0/γ=1 — sem nenhum DRL neural; os trabalhos com DRL moderno [L15, L18, L31, L37] não incluem métodos kernel/quantílicos. **O artigo SQR herda esse padrão**: cita RSLAQ/DDQN/GAN-DDQN mas não os avalia (campo `relation_to_our_gaps` de [L55]).

### 3.2 O par SQR × KBRL sob lupa (a comparação que reproduzimos)

| Aspecto | KBRL [L23]+10.1109/twc.2022.3195570 | SQR [L55]+10.1109/ccnc65079.2026.11366589 |
|---|---|---|
| Ambiente | Simulador Python próprio (traços ns-3 p/ fading; TR 36.942; 4 cenários 70–200 RBs; decisão 50 ms) | Simulador system-level próprio **não publicado** ("não é ns-3"); 2 GHz, 30 dBm, raio 2 km, 15 dBi |
| Granularidade SLA | Por **slice** | Por **UE** (p5 do throughput/UE; 0,5/1,0/1,5 Mbps) |
| Estatística | **30 runs/algoritmo/cenário; IC 90–95%** | "múltiplas execuções" — seeds/duração/IC **não publicados** |
| Código | Público (github.com/jjalcaraz-upct/network-slicing) | Não publicado |
| Resultado claim | Violação "quase nula" com menos recursos que os baselines em todos os cenários | Violação <1%; economia de ">60%" (Sec. V) vs "20%" (Conclusão) — inconsistente |

**Implicação de comparabilidade**: no artigo original, SQR e KBRL foram avaliados em **simuladores diferentes** (o de Alcaraz reimplementado dentro do simulador não publicado do SQR) com **granularidades de SLA diferentes** — a comparação publicada não é reproduzível nem tem denominador comum (G7). Nossa campanha restaura as duas condições: port fiel do KBRL (commit 65a3891, repositório oficial público) e do SQR rodando **no mesmo** ns-3/5G-LENA, com métricas pré-declaradas nos dois níveis (per-UE e per-slice) e IC 95% por seed — resultado: 84,3–89,9% de violação para todos os métodos no cenário publicado (2 km/30 dBm), sem separação entre γ=0 e γ=1.

---

## 4. Comparação de ambientes de avaliação

### 4.1 As quatro classes de fidelidade presentes na amostra

1. **Física 3GPP completa em simulação (ns-3/5G-LENA, MATLAB 5G Toolbox)** — RLF, MCS, HARQ, scheduler por RBG modelados; calibração possível contra 3GPP [L45]+10.1016/j.simpat.2019.101933, [L47]+10.1016/j.simpat.2022.102580; canal configurável até ray-tracing site-specific [L43]+10.1109/ccnc65079.2026.11366300. É onde RLF de borda e link budget **são observáveis** — e a classe do nosso ambiente (MODA: ns-3.46 + 5G-LENA v5.2, C efetivo = 53 PRBs medido em telemetria).
2. **Emulação/testbed com RF real ou semi-real (Colosseum, OAI RFsim, srsRAN, USRP)** — fidelidade de stack máximo, escala mínima; saturação por CPU é limitação declarada dos próprios autores [L15]+10.1109/iccworkshops67674.2025.11162144 (máx. 3 UEs simultâneos por saturação de ZeroMQ); regime de congestionamento deliberado estudado em RFsim [L37]+10.1109/mecom67453.2025.11439664.
3. **Simuladores próprios não calibrados** — KBRL [L23] (o mais bem documentado da classe: traços ns-3 + TR 36.942 + MCS com BLER-alvo), GAN-DDQN [L30], SafeSlice [L25] ("we do not have full access to real O-RAN interactions" — domain randomization), SQR [L55] (o menos documentado: simulador não publicado; parâmetros que o tornariam replicável ausentes — G2). Nenhum modela RLF (nosso diagnóstico atribuiu 56,6% das janelas com demanda a entrega nula por RLF de borda — fenômeno invisível nesta classe).
4. **Plataforma não nomeada** — 17/60; inclui abstract-only de periódicos fechados [L17, L26, L29] e omissão de autores [L34, L35, L36].

### 4.2 Integração O-RAN real (E2/RIC) — quem de fato validou o quê (G5)

| Trabalho | Canal E2 | O que mede | Paridade IPC×E2? |
|---|---|---|---|
| ns-O-RAN [L06]+10.1145/3592149.3592161 | ASN.1 KPM+RC → RIC OSC/OpenRAN Gym | Estabilidade funcional (medição única, sem repetições) | Não |
| NORI [L12]+10.14209/sbrt.2025.1571157244 | E2AP v2.02.03 + E2SM-KPM v2.0.3, RIC OSC Release I | Validação qualitativa de alocação de PRB via xApp exemplo | Não |
| FlexRIC [L11]+10.1145/3485983.3494870 | SDK; RTT/2, −83% CPU vs referência OSC | Latência/CPU do RIC | Não |
| AI-inside-RIC [L14]+10.1109/icccn69946.2026.11662852 | OAI RFsim + RIC | **Latência de inferência dentro do RIC** (LR 1–5 µs; serviço fim-a-fim 0,5–4 ms; >95% <10 ms) | Não |
| REAL [L15]+10.1109/iccworkshops67674.2025.11162144 | SDR leve (ZeroMQ) | Throughput por UE via E2; CDFs QoS | Não |
| ORANSlice [L16]+10.1145/3636534.3701544 | Plataforma física (Arena/X5G) | Demonstração de plataforma (sem comparativo) | Não |
| xDevSM [L20]+10.1109/tnsm.2026.3733344 | Multi-stack (OAI/srsRAN/X5G) | Latência de xApp portátil (4,5–7 ms) | Não |
| DRL slicing Colosseum [L31]+10.1109/icnc59896.2024.10556357 | KPMs via OpenRAN Gym em SRNs | Taxa de violação; PRBs consumidos | Não |
| **Nosso xApp SQR (Fase 5)** | e2sim E2AP v3.01 + E2SM-KPM + RIC Control via NORI | **Paridade bit a bit IPC×E2 (3585/3585 decisões) + latência p50 2,39 ms / p95 9,28 ms** | **Sim — não encontrado na literatura da amostra** |

A latência que medimos é compatível com a ordem de grandeza publicada: [L14] reporta serviço fim-a-fim 0,5–4 ms (transporte E2 ~1,5–2 ms dominante) e [L20] 4,5–7 ms — nosso p95 de 9,28 ms com budget de 50 ms é consistente com a classe. Nenhuma entrada da matriz reporta **comparação de paridade entre caminho de controle local (IPC) e caminho E2** — a metodologia de validação que usamos (decisões idênticas bit a bit nos dois caminhos) não tem antecedente nas 60 entradas; registrada como observação "não encontrada na amostra", não como inexistência absoluta (§7).

---

## 5. Comparação de suporte estatístico

Campo `statistical_method` das 60 entradas, classificado nesta sessão:

| Nível de suporte estatístico | n | Entradas |
|---|---|---|
| **Seeds/runs explícitos + intervalo de confiança** | **1** | [L23]+10.1109/twc.2022.3195570 (30 runs/algoritmo/cenário; IC 90% treino, 95% inferência) |
| Estatística descritiva parcial, sem seeds/IC (CDFs, percentis, médias, ablação) | ~8 | L07 (CDFs/p40/p80), L08 (médias/CDFs), L14 (runs de 10 min; ablação de ruído; **sem IC**), L24 (melhores 64 runs — risco de seleção), L25 (5-fold CV do cost model; seeds não reportados), L27 (média móvel; média de 2 semanas de trace), L31 (~150 episódios), L49 (ferramenta SEM) |
| Declaração explícita de ausência ("sem seeds/runs/IC no texto") | 7 | L06, L12, L15, L16, L18, L20, L55 |
| "Não reportado" (majoritariamente abstract-only) | ~43 | demais |
| n/a (teoria/survey) | 2 | L02, L13, L21 |

**Achado quantitativo central (generaliza G2/G4 para o campo)**: apenas **1 de 60** trabalhos incluídos — exatamente o KBRL, baseline do SQR — publica número de runs e intervalos de confiança. O artigo SQR [L55] alega "múltiplas execuções para confiabilidade estatística" sem publicar seeds, duração ou IC. Nosso protocolo (10 seeds × 4 métodos × 600 s, IC 95% por bootstrap na unidade seed, métrica primária pré-declarada) iguala o padrão de [L23] e supera a prática observada da amostra. Corolário metodológico: [L24] reporta "melhores 64 runs" por abordagem — seleção pós-hoc que inflanta estimativas; nenhum revisor externo pode detectar isso sem o número total de runs, que não é publicado.

---

## 6. O que é comparável e o que não é

### 6.1 Regras extraídas da matriz

| Comparação | Comparável? | Justificativa com evidência |
|---|---|---|
| Δ relativo **dentro** de um mesmo experimento (método vs baseline no mesmo ambiente/run) | ✅ Sim | É o único desenho que a amostra inteira suporta; ex.: [L31] 8,3× menos violação vs DQN no mesmo Colosseum; [L25] −83,23% latência vs seus baselines no mesmo guided-DR |
| Números absolutos **entre** papers (taxa de violação, % SLA) | ❌ Não | Ambientes e definições divergem: violação <1% no simulador próprio do SQR [L55]; >95% de confiabilidade em MATLAB [L18]; reduções de 83–93% em guided-DR [L25]; 84–90% de violação na nossa física ns-3 — quantidades sem denominador, canal, RLF ou tráfego comuns |
| Métricas de SLA entre papers | ❌ Não sem harmonização | Per-UE p5 [L55] × per-slice satisfação [L23, L26] × latência de cauda/URLLC [L15, L28, L37] × CVaR₀,05 [L28] — granularidades que não se convertem sem recálculo (G7) |
| Eficiência de espectro ("economia de RBs") entre papers | ❌ Não | Denominadores não declarados: ">60%" vs "20%" no próprio SQR [L55] (G4); PRB ratio em [L07] usa CDF por slice sem agregação única |
| Latência de loop E2 entre papers | ⚠️ Ordinal apenas | Ordens de grandeza comparáveis (µs–ms) mas stacks/hardware distintos: [L11] RTT/2 em SDK dedicado; [L14] OAI RFsim em 1 máquina; [L20] multi-stack; nosso e2sim local DEGRADED — nenhum substitui medição no próprio setup |
| Reprodução de baseline publicado | ✅ Condicionada ao código | Só possível quando há artefato público: KBRL [L23] sim; SQR [L55] não (G2); SafeSlice/TL-DRL publicam repo citado, não inspecionado nesta sessão [L24, L25] |
| Calibração física (potência/raio/antena/LOS) entre papers | ⚠️ Via 3GPP como referência comum | [L45]+10.1016/j.simpat.2019.101933 e [L47]+10.1016/j.simpat.2022.102580 estabelecem o método (curvas 3GPP/ITU); cenários de controladores raramente declaram essa âncora — 2 km/30 dBm/15 dBi do SQR [L55] viola link budget de macro (nosso déficit medido 25–30 dB; corroboração externa em [L54]+10.1186/s13638-022-02133-3) |

### 6.2 Consequência para o nosso artigo

As únicas comparações que podemos publicar como evidência são: (a) SQR γ=0 × γ=1 × KBRL × equal **no nosso ambiente**, com IC 95%; (b) física vs parâmetros publicados via **sweep controlado** (só o raio muda: 0% @300 m → 88% @2 km); (c) paridade IPC×E2 do nosso xApp. Qualquer confronto numérico direto com [L55], [L18], [L25] etc. é invalidado pela tabela 6.1 — e deve ser apresentado como "não comparável quantitativamente", citando esta seção.

---

## 7. Limitações da busca (fonte: `tools/search-log.jsonl`)

O log contém **137 entradas** registradas em **um único dia** (2026-10-18; campo `ts` de todas as entradas) — 132 da sessão de busca original + 5 da sessão de revisão (2 Crossref, 1 arXiv API, 2 recontagens/verificações locais). Distribuição por fonte: openalex 45, crossref 22, websearch 17, webfetch 10, datacite 8, arxiv-api/arxiv 7, screening 4, crossref+datacite 4, local 5, demais (semanticscholar, webreader, springer, curl+pdftotext etc.) 15.

Limitações identificadas, cada uma com evidência no log:

1. **Cobertura de índice (lag) do OpenAlex**: as 31 queries OpenAlex somam **89.535 resultados brutos**, mas a API retornou **top-N por relevância (N=5–25 por query** — campo `returned`)**.** Saturação foi assumida pela estratégia (§5 de `search-strategy.md`), não demonstrada: queries com milhares de hits (ex.: 29.773 na C3; 5.326 na B1) foram inspecionadas só na cabeça da distribuição. Papers relevantes fora do top-N por ranking de relevância não podem ser excluídos.
2. **Papers IEEE muito recentes não indexados**: o artigo objeto [L55]+10.1109/ccnc65079.2026.11366589 **não veio de busca** — veio do workspace (PDF + tradução verificada); sua descoberta dependeu do Crossref/WebSearch (log 41–47, 77–96). O mesmo vale para entradas 2026 (L14, L18, L20, L28, L43, L48), várias localizadas por WebSearch/crossref_check, não por OpenAlex. A busca sistemática subestima sistematicamente o ano corrente.
3. **Somente inglês** (declarado em `search-strategy.md` §2/§6); literatura em outros idiomas não coberta.
4. **Snowballing parcial**: forward citation check feito sobre **12 incluídos** inspecionados no OpenAlex (registrado nas razões de exclusão do `screening-log.csv` como "citação por incluído não verificada (OpenAlex, 12 incluídos)"); backward snowballing via Crossref registrado no log (entrada 60, `snowball_doi_search`). Snowballing de segunda ordem não foi executado.
5. **Contagens de WebSearch aproximadas e manuais**: entradas websearch do log registram contagens estimadas ("~8 resultados relevantes") sem raw JSON — não auditáveis como as OpenAlex.
6. **Entradas de verificação com resultado vazio**: várias chamadas crossref/datacite retornaram `results: None` (log 41–49, 52–64) — foram tentativas de verificação de DOI/título, não buscas de descoberta; não inflacionam a cobertura.
7. **33/60 entradas só com abstract+metadados** (§1) **[CORRIGIDO: era 29/60]**: campos "não reportado" dessas linhas podem estar presentes em full-text paywalled — **ausência no abstract não é omissão do autor**. Nenhum gap declarado em `research-gaps.md` repousa exclusivamente em linha abstract-only sem corroboração full-text ou evidência própria.
8. **Janela temporal única**: toda a busca em 1 dia; sem replicação longitudinal (novas publicações entre a busca e a submissão do artigo não serão capturadas — reexecutar queries D1–D5 e B5 próximo à submissão).
9. **Viés de linhagem declarado**: as queries dos eixos B/D podem recuperar o próprio SQR e o grupo Tuerxun/Nakao (L03, L55); o relacionamento está registrado na matriz para julgamento de independência.

---

## 8. Síntese

1. O estado da arte avalia controladores de slice em **quatro classes de fidelidade** que não trocam números entre si (§4.1); a classe ns-3/5G-LENA é a maior e a única onde RLF/link budget são observáveis — e é a nossa.
2. A combinação kernel+quantílico sem rede neural tem 1 antecedente (KBRL [L23]) e 1 instanciação (SQR [L55]); os primos neurais distributional [L30, L60, L61] e o RL restrito moderno [L24–L29, L56–L59] são os comparadores ausentes do artigo SQR (G8).
3. O suporte estatístico do campo é deficitário: 1/60 com seeds+IC (§5). Nosso protocolo iguala o melhor da amostra.
4. A validação E2 real existe na literatura (8 entradas, §4.2), mas nenhuma medida **paridade de caminho de controle** (IPC×E2) — diferencial do nosso xApp.
5. As comparações publicáveis estão delimitadas na §6.2; confrontos numéricos cross-paper são metodologicamente inválidos e devem ser reportados como tais.

---

*Próximo artefato: `research-gaps.md` (6/6) — gaps cruzados com G1–G8, cada um com evidência da matriz e pergunta testável. Nenhum manuscrito será escrito antes da revisão e aprovação explícita do pesquisador (regra da skill).*
