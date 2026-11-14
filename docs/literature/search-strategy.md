# Estratégia de busca — revisão de literatura (O-RAN / RAN slicing / RL / simulação)

> **Skill:** oran-literature-review · **Artefato:** 2/6 (`search-strategy.md`) · **Data:** 2026-10-18
> **Insumo:** `work-understanding.md` (mesmo diretório) — gaps G1–G8 e as 8 perguntas de posicionamento (§7).
> **Sintaxe validada nesta sessão** (`docs/literature/tools/search-log.jsonl`, entradas `20261018T155249` e `20261018T155706`): a API OpenAcept aceita `AND`/`OR` em caixa alta, frases entre aspas duplas e filtro `from_publication_date` — as queries abaixo foram escritas nessa sintaxe. 1ª probe: `"O-RAN" AND ("near-RT RIC" OR xApp) AND "resource allocation"` (891 resultados, 2020+); 2ª probe (eixo B): `"network slicing" AND ("deep reinforcement learning" OR "safe reinforcement learning")` (5.326 resultados, 2019+).

---

## 1. Questão de pesquisa

> **O que a literatura já fez** em (a) validação real de controladores O-RAN (E2/RIC/xApp), (b) RAN slicing com aprendizado por reforço, (c) avaliação reprodutível e calibrada em simuladores ns-3/5G-LENA com SLA, e (d) algoritmos safe/kernel/quantile RL e seus baselines — **para responder às 8 perguntas de posicionamento** de `work-understanding.md` §7 e declarar a novidade do artigo proposto ("auditoria empírica, calibração realista e validação O-RAN de controladores quantílicos") apenas contra o que estiver demonstrado que existe.

## 2. Fontes de busca

| Fonte | Papel | Registro |
|---|---|---|
| **OpenAlex API** — `python3 docs/literature/tools/openalex_search.py "<QUERY>" --from-year 2019 --out <prefixo>` | Fonte primária (título+abstract+fulltext indexado; citações para snowballing) | automático em `tools/search-log.jsonl` + JSON bruto em `tools/raw/` |
| **WebSearch** (tool da sessão) | Complementar: literatura cinzenta, artefatos (GitHub/DOIs quebrados), papers muito recentes ainda não indexados (ex.: CCNC 2026), specs O-RAN Alliance/3GPP | registro **manual** (append em `tools/search-log.jsonl` com `"source": "websearch"`, data e contagem) |
| **Crossref** — `python3 docs/literature/tools/crossref_check.py` | Verificação de DOI/metadados de todo incluído (nunca confiar em DOI lembrado) | saída anotada no `screening-log.csv` |

**Período:** ≥ 2019-01-01 (filtro aplicado em toda query OpenAlex). Justificativa: a literatura de xApps/RIC ganhou corpo após a fundação da O-RAN Alliance (2018) e o pico de slicing+DRL é 2019+. **Exceção controlada:** trabalhos seminais pré-2019 (ex.: KBRL original de Alcaraz et al., lineage GP-safe-RL) podem entrar **somente via snowballing reverso** a partir dos incluídos, com justificativa registrada no `screening-log.csv`.

**Idioma:** inglês (idioma da área; queries em inglês). Não há busca em português/espanhol — limitação declarada.

## 3. Eixos e queries booleanas (inglês)

Cada query é executada no OpenAlex com `--from-year 2019`; resultados ordenados por relevância (padrão da API), top-N inspecionados (N≥50 ou saturação de relevância). Sinônimos foram expandidos antes de qualquer conclusão de lacuna (regra da skill: **poucos resultados ≠ gap**).

### Eixo A — Arquitetura O-RAN: RIC, xApps, interface E2 (→ perguntas 5 e parte da 2)

| ID | Query |
|---|---|
| A1 | `"O-RAN" AND ("near-RT RIC" OR "RAN intelligent controller") AND ("xApp" OR "rApp")` |
| A2 | `("E2 interface" OR "E2AP" OR "E2SM") AND ("xApp" OR "near-RT RIC") AND (simulation OR emulator OR "testbed")` |
| A3 | `("ns-O-RAN" OR "ns-O-RAN gym") OR ("ns-3" AND "O-RAN" AND ("E2" OR RIC))` |
| A4 | `"O-RAN" AND ("closed loop" OR "control loop") AND (latency OR overhead) AND ("RAN control" OR "near-real-time")` |
| A5 | `"O-RAN" AND "network slicing" AND ("xApp" OR RIC) AND (SLA OR "service level agreement" OR KPM)` |

*Objetivo: quem já validou xApps/loops E2 reais (não apenas arquiteturas conceituais), com que métricas de latência/paridade; linhagem NORI/ns-O-RAN/FlexRIC.*

### Eixo B — RAN/network slicing + aprendizado por reforço (→ perguntas 1, 4, 6, 8)

| ID | Query |
|---|---|
| B1 | `"network slicing" AND ("reinforcement learning" OR "deep reinforcement learning") AND ("RAN" OR "radio access network") AND ("resource block" OR PRB OR scheduler)` |
| B2 | `"RAN slicing" AND ("safe reinforcement learning" OR "constrained reinforcement learning" OR "safe RL" OR "risk-aware")` |
| B3 | `"network slicing" AND (SLA OR "service level agreement") AND "reinforcement learning" AND (violation OR guarantee)` |
| B4 | `("multi-agent reinforcement learning" OR "distributed reinforcement learning") AND "RAN slicing" AND "resource allocation"` |
| B5 | `"network slicing" AND ("quantile" OR "distributional" OR uncertainty) AND "reinforcement learning"` |

*Objetivo: SOTA e baselines de slicing+RL; se safe/constrained/quantile já foi aplicado a slicing; suporte estatístico típico (seeds/IC); tratamento de granularidade per-UE vs per-slice; caracterização de regimes de carga.*

### Eixo C — Simulação ns-3/5G-LENA, reprodutibilidade, SLA e calibração (→ perguntas 1, 2, 3, 4)

| ID | Query |
|---|---|
| C1 | `"ns-3" AND ("5G-LENA" OR "NR module" OR "CTTC") AND (slicing OR scheduler OR "resource allocation")` |
| C2 | `("5G-LENA" OR "ns-3") AND (reproducibility OR replicability OR "open science" OR "artifact evaluation")` |
| C3 | `(ns-3 OR "5G-LENA" OR "system-level simulation") AND (calibration OR "link budget" OR "path loss model") AND validity` |
| C4 | `(ns-3 OR "5G-LENA") AND ("network slicing" OR "RAN slicing") AND (SLA OR throughput) AND (evaluation OR benchmark)` |
| C5 | `"system-level simulation" AND 5G AND ("radio link failure" OR RLF OR "cell edge") AND (coverage OR feasibility)` |

*Objetivo: precedentes de avaliação de slicing em ns-3/5G-LENA; práticas de calibração (potência/raio/antena/LOS — G1); instrumentação de ações no MAC (G3); protocolos de reprodutibilidade (G2); estudo de RLF/borda (§4.3 do entendimento).*

### Eixo D — Algoritmos: kernel/quantile/GP-safe RL × baselines (→ perguntas 8 e parte da 6)

| ID | Query |
|---|---|
| D1 | `("kernel-based reinforcement learning" OR KBRL) AND (wireless OR RAN OR "network slicing")` |
| D2 | `("Gaussian process" OR "kernel regression") AND "reinforcement learning" AND (safety OR "safe exploration" OR uncertainty)` |
| D3 | `"quantile regression" AND "reinforcement learning" AND (risk OR uncertainty OR "distributional")` |
| D4 | `("constrained MDP" OR CMDP OR "safety layer") AND ("resource allocation" OR scheduling) AND (wireless OR 5G)` |
| D5 | `(feasibility OR projection) AND ("safe reinforcement learning") AND (action OR "control barrier" OR "resource allocation")` |

*Objetivo: linhagem KBRL (o baseline do artigo) e sua posição na literatura; se GP/kernel-safe-RL e quantile/distributional RL já foram combinados para alocação de rádio (núcleo do SQR); análise de projeção de ações (fallback do SQR-JC, G8).*

**Total: 19 queries** (A1–A5, B1–B5, C1–C5, D1–D5). Toda execução (OpenAlex ou WebSearch) é registrada com data, contagem e fonte; exclusões vão ao `screening-log.csv` com razão; incluídos vão à `literature-matrix.csv`.

## 4. Critérios de elegibilidade

### 4.1 Inclusão (todas devem valer)

- **I1 — Período:** publicação ≥ 2019 (exceção seminal pré-2019 só via snowballing, justificada).
- **I2 — Tipo:** revisado por pares (journal/conferência/workshop) OU preprint com artefatos verificáveis (código/dados públicos, ex.: arXiv + repositório). Documentos normativos (O-RAN Alliance WGs, 3GPP TS/TR, RFC) entram como **fonte normativa** em campo próprio da matriz, não como evidência de estado da arte acadêmico.
- **I3 — Relevância mínima (pelo abstract):** aborda ≥1 dos eixos A–D com componente mensurável (arquitetura implementada, algoritmo avaliado, simulador validado, protocolo proposto). Papers puramente posicionais (visões/editoriais) só entram se definirem agenda citada pelos incluídos.
- **I4 — Verificabilidade:** DOI/metadados confirmados no Crossref; texto completo obtido para entrada na matriz (abstract-only ⇒ registro no `screening-log.csv` como "pending-fulltext", sem inferir achados).
- **I5 — Relação com os gaps:** conecta-se a ≥1 das perguntas de posicionamento 1–8 (`work-understanding.md` §7) — inclusive como antecedente que enfraquece nossa novidade (mais importante registrar do que omitir).

### 4.2 Exclusão (qualquer uma basta; cada exclusão é logada com razão)

- **E1 — Sem componente RAN:** slicing apenas de core/E2E sem controle de rádio comparável; redes ópticas/SDN sem camada física sem fio.
- **E2 — RL fora de redes sem fio:** aplicações de safe/distributional RL em robótica, jogos, finanças etc. — exceto se forem a origem de um mecanismo citado por trabalho incluído (entram como "antecedente metodológico" via snowballing, razão registrada).
- **E3 — Sem revisão por pares e sem artefatos:** white papers de vendors, blogs, relatórios técnicos não auditáveis (specs normativas seguem a regra de I2).
- **E4 — Duplicata:** mesmo DOI; ou título+autores normalizados idênticos (pré-print vs versão final: mantém-se a versão publicada, pré-print logado).
- **E5 — Sem conteúdo avaliável:** abstract de palestra chave, resumo de painel, erratum.
- **E6 — Fora de escopo após full-text:** menciona eixo no abstract mas o corpo não contribui para nenhuma pergunta 1–8 (razão específica registrada).
- **E7 — Não recuperável:** texto completo inacessível após esgotar canais (DOI, repositório do autor, pedido direto) ⇒ permanece no `screening-log.csv` como "unavailable-fulltext"; **não** entra na matriz com dados inferidos.

### 4.3 Regras anti-vieses (da skill e do `GUIA` §12)

1. **Poucos resultados ≠ lacuna científica:** antes de declarar ausência, expandir sinônimos (lista de variantes por eixo: `RIC/xApp/rApp/near-RT/non-RT`; `slicing/network slicing/RAN slicing/slice admission`; `safe/constrained/risk-aware/conservative`; `quantile/distributional/CVaR`; `ns-3/5G-LENA/NR module/NS3`), rodar snowballing para trás (referências) e para frente (citado por) sobre os incluídos, e inspecionar o related work de ≥3 incluídos por eixo.
2. **Claim de autor ≠ fato verificado:** a matriz registra o que o paper *alega* e o que tem *artefato verificável*; campos não reportados no texto = "não reportado".
3. **Nunca inventar citação/DOI/métrica**; todo DOI incluído passa pelo `crossref_check.py`.

## 5. Fluxo de screening e extração

```
query (OpenAlex/WebSearch) → log automático (search-log.jsonl)
  → dedup (DOI → título normalizado)
  → triagem por abstract vs I3/E1–E5 → screening-log.csv (decisão + razão + data)
  → full-text dos candidatos → I4/I5/E6/E7
  → extração na literature-matrix.csv (18 campos: problema, arquitetura, algoritmo,
    state/action/reward, ambiente, tráfego/topologia, baselines, métricas, estatística,
    resultados, limitações, reprodutibilidade, relação com G1–G8, verificado)
  → snowballing (backward refs + forward cited-by no OpenAlex) sobre os incluídos → novas queries se conceito novo emergir
```

**Ordem de leitura e corte:** dentro de cada eixo, priorizar por (a) presença de artefato público, (b) citações, (c) recência — sem descartar trabalhos pouco citados devenues relevantes via snowballing. Meta de cobertura: saturação (novas queries param de trazer candidatos elegíveis).

## 6. Limitações declaradas da estratégia

- **Cobertura do OpenAlex:** indexação orientada a título+abstract+referências; papers de conferências IEEE muito recentes (ex.: CCNC 2026) podem ainda não estar indexados — por isso o artigo-base SQR (DOI 10.1109/CCNC65079.2026.11366589) é verificado **diretamente no Crossref** e seus citados via WebSearch.
- **Idioma:** queries somente em inglês; literatura publicada só em outros idiomas não é coberta (limitação aceita — o domínio publica em inglês).
- **Grey literature:** specs da O-RAN Alliance e do 3GPP são localizadas via WebSearch e usadas como fontes normativas, não como evidência de SOTA acadêmico.
- **Viés de confirmação:** as queries dos eixos B/D podem recuperar o próprio SQR e trabalhos do mesmo grupo; registrar relacionamento (mesma linhagem/autores) na matriz para julgar independência.
- **O que esta estratégia NÃO decide:** a existência de gap científico. Isso só é declarado em `research-gaps.md` após matriz completa + snowballing + aprovação do pesquisador.
