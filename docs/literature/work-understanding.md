# Entendimento do trabalho — síntese para posicionamento da revisão de literatura

> **Skill:** oran-literature-review · **Artefato:** 1/6 (`work-understanding.md`) · **Data:** 2026-10-18
> **Fontes lidas nesta sessão:** `README.md` (§1–§4), `AGENTS.md` (§1–§8, §10), `GUIA_IMPLEMENTACAO_SQR_O_RAN_PYTHON_NS3.md` (§0–§1, §12), `MODA/docs/sqr/assumptions_register.md` (v1.3, R01–R31), `MODA/docs/sqr/fase5.4-xapp-sqr-paridade.md`, `MODA/ns-o-ran-gym/results/sqr_campaign/tables/summary_by_method.csv` (lido integralmente nesta sessão).
> **Regras aplicadas:** claim de autor ≠ fato verificado por nós; valores ausentes = "não reportado"; lacuna de busca ≠ lacuna científica (ver §8).

---

## 1. O que este trabalho é

Reprodução e extensão do artigo **SQR** — Tuerxun & Nakao, *"Safe RAN Slicing in O-RAN: Minimizing SLA Violations via Model-Based Reinforcement Learning"*, IEEE CCNC 2026, DOI [10.1109/CCNC65079.2026.11366589](https://doi.org/10.1109/CCNC65079.2026.11366589) — no ambiente **MODA** (ns-3.46-dev + 5G-LENA v5.2 + Python/ns-o-ran-gym), com baseline **KBRL** (Alcaraz et al., IEEE TWC 2022; referenciado no workspace como [jjalcaraz-upct/network-slicing](https://github.com/jjalcaraz-upct/network-slicing) — DOI a confirmar na fase de busca) e integração O-RAN real via **NORI** (lasseufpa/nori).

A reprodução rigorosa **falhou em replicar o resultado central do artigo por razões físicas identificáveis** e, com isso, o projeto evoluiu para uma **auditoria empírica do estado da arte** que fundamenta um artigo novo (`README.md` §3–§4): *"Safe RAN slicing sob incerteza de viabilidade: auditoria empírica, calibração realista e validação O-RAN de controladores quantílicos"*.

## 2. O problema

**Formulação publicada pelo artigo** (`GUIA` §1.1, transcrição do artigo Seção III): uma estação-base reparte `C` PRBs entre `L` slices; a cada período de decisão `t` (100 ms), cada agente `i` observa estado local `s_t^(i)` + visão global `s̄_t`, propõe ação inteira `a_t^(i) ∈ {0..C}` com `Σ_i a_t^(i) ≤ C`. Objetivo: **minimizar a utilização média de PRBs sujeito à taxa de violação de SLA por slice ≤ δ** (exemplo publicado: δ = 1%), com `y_t^(i)` exemplificado como o 5º percentil do throughput por UE e SLA per-UE de 0,5/1,0/1,5 Mbps em 3 slices eMBB.

**Problema de pesquisa do nosso artigo novo** (`GUIA` §0, "Definição do problema em uma frase"): o SQR aprende risco por slice e corrige excesso de propostas com projeção global proporcional; investigamos **quando isso deixa de preservar SLAs** (regimes de carga e de viabilidade física) e se coordenação conjunta consciente de risco (SQR-JC, `GUIA` §8) reduz violações sem inflar consumo.

**Critério central de validade** (`GUIA` §0): reportar o **trade-off risco × consumo**, incluindo casos fisicamente impossíveis — nunca "menos violações" ao custo de reservar tudo, nem "menos PRBs" degradando indiscriminadamente.

## 3. O método

- **SQR** (*Safe Quantile Regression*, port fiel nosso, sem rede neural): regressor quantílico online **kernel Matérn** (ν=1,5 normalizado, R08) com memória de vetores de suporte B=2000 `[ARTIGO, Seção V]` e poda por influência `[ASSUMIDO, R10]`; política hierárquica com margem de segurança `σ` e dois modos (confiante via f_τ; incerto via ε-softmax `[ARTIGO, Tabela I: ε=0,1]` ou k-NN safety-first K=10 `[ARTIGO, Tabela I]` — leitura emendada R21: filtrar sucessos antes de tomar os K vizinhos); coordenação por **projeção proporcional inteira** `[ASSUMIDO, R24]`; adaptação de τ∈[0,01;0,05] e M `[faixa ARTIGO, dinâmica ASSUMIDO, R11]`; atualização virtual pós-projeção com peso negativo pequeno `[ARTIGO qualitativo, valor ASSUMIDO, R12]`.
- **Variantes:** SQR γ=0 (sem custo na seleção) × SQR γ=1 (com `f_τ(z,a) − γ·a/C`); a penalização de falha crítica `P_penalty=2` **permanece ativa nas duas** (`GUIA` §1.3; AGENTS §5.7; R25) — os dois mecanismos de custo são distintos e confundi-los invalidaria a comparação das Figs. 2–4.
- **Baselines:** KBRL (port fiel do repositório oficial, Fase 3) e **equal** (partição estática C/3).
- **Ambiente:** ns-3.46-dev + 5G-LENA v5.2 com scheduler novo `SqrMacScheduler` (cota **absoluta** de RBGs por slice, RBGs ociosos preservados — R27); loop de controle Python↔ns-3 de 100 ms via IPC (semáforos POSIX + CSV, AGENTS §4) e via **E2/RIC** (NORI, Fase 5); cenário "fiel à Tabela I" (1 gNB, 2 GHz, 10 MHz → **C efetivo = 53 PRBs** medido na telemetria, R01; 30 dBm; antena omni 15 dBi, R30; UMa; Poisson 2 UE/min, sessão exp. 60 s; UDP greedy 100 Mbps/UE, R04).
- **Estatística:** campanha pareada 4 métodos × 10 seeds × 600 s, IC 95% por bootstrap na unidade seed, métricas pré-declaradas nos dois níveis (per-UE e per-slice, R19/R26/R23).

## 4. Evidências próprias da equipe (números verificados)

### 4.1 Campanha central — 40 runs (4 métodos × 10 seeds × 600 s), raio 2 km / 30 dBm (parâmetros publicados do artigo)

Violação de SLA per-UE (métrica primária, R19) — **verificada nesta sessão em `MODA/ns-o-ran-gym/results/sqr_campaign/tables/summary_by_method.csv`** (métrica `ue_time_violation_pct`, n_seeds=10 em todas as linhas):

| Método | Violação per-UE (IC 95%) | Violação per-slice (IC 95%) |
|---|---|---|
| equal | 84,27% [80,37–88,04] | 84,93% [80,06–89,53] |
| KBRL | 88,89% [86,82–90,90] | 89,57% [86,70–92,29] |
| SQR γ=0 | 89,64% [86,53–92,20] | 89,40% [86,36–92,13] |
| SQR γ=1 | 89,94% [86,74–92,50] | 89,76% [86,73–92,42] |

**Todos os métodos violam massivamente, incluindo o baseline sem agente** — os ICs de γ=0 e γ=1 se sobrepõem quase inteiramente. Consumo: `rb_cell_total_executed_mean` = 53,0 PRBs (equal, célula toda usada) vs 28,0 (SQR γ=0) e 27,4 (SQR γ=1) — o mecanismo de cota economiza RBs, mas a violação não cai. Consistente com `README.md` §3.1.

### 4.2 Sweep de raio — o discriminador (mesmo código, só o raio muda; network-only, 240 s, seed 1)

`README.md` §3.2 (tabela transcrita abaixo; resultado interno, não publicado):

| Raio | Eventos RLF | Violação SLA | Throughput mediano/UE |
|---|---|---|---|
| 300 m | 0 | **0,0%** | 7,3 Mbps |
| 800 m | 0 | 9,7% | 2,8 Mbps |
| 1500 m | 10 | 58,7% | 0,86 Mbps |
| 2000 m (artigo) | 36 | **88,0%** | 0,0 Mbps |

**Interpretação (nossa):** a implementação está **autovalidada** — 0% de violação quando a física permite, cadeia MAC→RLC→KPM→métrica correta. A inviabilidade é do **parâmetro publicado** (2 km/30 dBm), não do código.

### 4.3 Diagnóstico de causa raiz

`README.md` §3.1 + R28 do registro: **56,6% das janelas com demanda têm entrega nula**; UEs de borda em loop de RLF (~**2,7 quedas RLF/UE/600 s**; buffer RLC zerado a cada reestabelecimento; até **26 trocas de RNTI/UE em 600 s** observadas no soak). O artigo não modela nem discute RLF (R28).

### 4.4 Análise de link budget — déficit de 25–30 dB

`README.md` §3.3 (cálculo nosso, conservador): (1) antena 1×1 omni vs array 8×8 dual-pol (~15–20 dB); (2) UMa NLOS para todos (PL@2 km ≈ 148 dB vs ≈139,4 dB de TS 36.942 citado pelo artigo, ~9 dB); (3) 30 dBm vs 43 dBm/10 MHz de macro real (TR 38.104, ~13 dB); (4) TDD 3DL/7 (~43% DL) vs "TTI 1 ms" sugestivo de DL cheio (~2,3×). Itens 1+2+3 ≈ **25–30 dB**, a ordem de grandeza que encolhe o raio de viabilidade de ~2 km (macro real) para ~0,8–1,2 km (medido). Conclusão nossa: a Tabela I do artigo descreve uma *small cell* servindo uma macro-célula.

### 4.5 Gate R27 — aplicação real da cota no MAC (o que o artigo não demonstra)

Validação registrada em R27 (`assumptions_register.md`), ratificada 12/10/2026 e implementada como `SqrMacScheduler` em `scratch/sqr/`: smoke determinístico com fases Σa=0 / 0,5·C / C (300 janelas) — (i) utilização prb_scheduled/cota = 1,0000 nas 3 slices em Σa=C; (ii) em Σa=0,5·C, mediana de 57% dos RBGs ociosa, Σ sched ≤ (0,5·C+L)·slots em 100/100 janelas; (iii) Σa=0 ⇒ 0 RBG servido em 100/100 janelas; (iv) `scratch/rslaq` intocado. Telemetria por janela: `prb_requested/executed/scheduled/idle`. **Sem esse gate, γ·a/C é inobservável e a Fig. 4 do artigo não significa nada** (G3). Na campanha, `rb_cell_total_idle_est_mean` ≈ 75–77 PRBs ociosos/slot sob SQR vs ≈0 sob equal — evidência de cota aplicada em regime real.

### 4.6 Paridade IPC × E2 (Fase 5, modo DEGRADED)

`MODA/docs/sqr/fase5.4-xapp-sqr-paridade.md` (números lidos nesta sessão): paridade **bit a bit** IPC vs caminho E2 (e2sim E2AP v3.01 + adaptador `sqr-e2-adapter` + xApp SQR Python) em 120 s, seed 1, γ=1 — **3585/3585 decisões idênticas** (timestamp_ms, mode, raw_action, action, prb_requested, prb_executed; 0 divergências), **1199/1199 janelas** de telemetria ns-3 idênticas, 1213 linhas de KPM idênticas, 994 = 994 violações per-UE-janela, **0 timeouts**. Latência E2 (indication→controle, fim a fim, n=1195): **p50 2,39 ms / p95 9,28 ms / máx 12,05 ms** — dentro do orçamento de 50 ms do contrato com folga ≫5×; custo do caminho E2 sobre IPC: +1,23 ms (p50). Smoke do xApp: 299/299 indicações aplicadas. **Limitação declarada: modo DEGRADED** (e2sim local, sem Near-RT RIC real em k8s; RIC real é decisão pendente do usuário — AGENTS §8).

### 4.7 Regime inviável — fenômeno não descrito no artigo (base do G6)

`summary_by_method.csv` (lido nesta sessão): `modes_uncertain_group_pct` = **100% [100–100]** em SQR γ=0 e γ=1 (modo confiante 0%) — `A_safe` permanentemente vazio, τ saturado no piso, M no teto. O comportamento γ=0 × γ=1 do artigo só existe num **regime viável** que o artigo nunca caracteriza; nossa proposta: análise de regimes (viável × marginal × inviável) e diversidade de modos como métrica de diagnóstico.

### 4.8 Infraestrutura de rastreabilidade (diferencial de reprodutibilidade)

Registro de premissas v1.3 com **31 linhas** R01–R31, cada valor marcado `[ARTIGO, seção/tabela]` ou `[ASSUMIDO]` com justificativa e análise de sensibilidade; contrato de interface v1.2 (KPM estendidos + ação em cota absoluta); 12+ parâmetros essenciais do artigo não publicados e assumidos com justificativa (G2); 190 testes Python + smoke/soak com gates go/no-go; C lido da telemetria (53 PRBs @ 10 MHz μ=0), sem hardcode.

## 5. Claims do artigo vs. o que verificamos

| # | Claim do artigo (autor) | Status | Nossa evidência |
|---|---|---|---|
| A1 | "99% de SLA atendido" no cenário da Tabela I | **Não reproduzível no nosso ambiente** — contrariada | 84–90% de violação p/ todos os métodos (§4.1); 0% só @ 300 m (§4.2) |
| A2 | ">60% de economia de RBs" (Seção V, Fig. 4) vs "20%" (Conclusão) | **Internamente inconsistente**; sem denominador definido | G4; métricas nossas com denominadores explícitos e IC 95% |
| A3 | Arquitetura declarada "suitable for deployment in an O-RAN RIC" (Seção I, contribuições) | **Afirmado, não demonstrado** (sem E2/xApp real) | Nossa validação E2 com paridade bit a bit + latência (§4.6) |
| A4 | Simulador/código não publicados; C, banda, MCS, scheduler não reportados | Fato (omissão) | G2: 12+ parâmetros faltantes catalogados no registro |
| A5 | "Aplicação da cota Σa≤C no escalonamento" | **Não demonstrada** | Gate R27 com telemetria de RBGs ociosos (§4.5) |

*Nota de rastreabilidade: A1/A2/A3/A5 são transcrições do artigo via `GUIA` §1.2 e `README.md` §4; o artigo não recebeu (ainda) resposta dos autores às perguntas abertas do `GUIA` §15.*

**Correção (2026-10-18):** o rótulo anterior de A3 — "Compatível com near-RT RIC" — não consta no PDF do artigo (pdftotext: 0 ocorrências de "near-RT" e de "compat"; texto real: "We propose a novel MBRL agent architecture suitable for deployment in an O-RAN RIC", bullet de contribuições da Seção I) e foi substituído pela transcrição correta. A substância do gap (O-RAN apenas conceitual, sem demonstração E2) permanece — ver G5 na matriz ([L55], `relation_to_our_gaps`).

## 6. Gaps identificados (evidência nossa) — base do artigo novo

`README.md` §4, resumidos com sua evidência:

- **G1 — Inviabilidade física dos parâmetros publicados.** 30 dBm + 2 km + urban macro inconsistente com link budget (déficit 25–30 dB, §4.4); sweep de raio 0%→88% (§4.2).
- **G2 — Irreprodutibilidade por omissão.** 12+ parâmetros essenciais não publicados (C/banda/numerologia, tráfego, T_softmax, ν/ℓ do Matérn, regra de poda, adaptação τ/M, seeds/duração). Contribuição: protocolo `[ARTIGO]/[ASSUMIDO]` auditável (§4.8).
- **G3 — Sem evidência de aplicação real da cota no MAC.** Gate R27 (§4.5) — sem isso, a Fig. 4 do artigo é inobservável.
- **G4 — Percentuais internamente inconsistentes** (">60%" vs "20%", Claim A2). Contribuição: denominadores explícitos (UE-tempo × slice-tempo) + IC 95% por seed.
- **G5 — O-RAN apenas conceitual.** Nossa contribuição: xApp sobre e2sim/NORI com paridade bit a bit e latência p50/p95 (§4.6); RIC real pendente.
- **G6 — Regime operacional ignorado.** Sob inviabilidade, `A_safe` vazio 100% do tempo, γ=0 ≈ γ=1 (§4.1, §4.7); os efeitos do artigo existem só num regime viável nunca caracterizado.
- **G7 — Granularidade de SLA não harmonizada** (SQR per-UE × KBRL per-slice). Contribuição: reporte nos dois níveis com métrica primária pré-declarada (§4.1 mostra ambos).
- **G8 — Sem heterogeneidade nem DRL moderno.** Só 3 eMBB; RSLAQ/DDQN/GAN-DDQN citados, não avaliados. Herda-se do `GUIA`: eMBB+URLLC+mMTC, transições de carga, proposta SQR-JC (coordenação conjunta consciente de risco no lugar da projeção proporcional, que pode tornar ações localmente seguras em globalmente inseguras).

## 7. O que a revisão de literatura precisa responder

Perguntas de posicionamento, mapeadas aos gaps (o checklist de novidade do `GUIA` §12, última linha, exige explicitamente revisar: *safe RAN slicing*, *joint/constrained resource allocation*, *risk-aware coordination*, *distributional/quantile RL* e *feasibility projection* — "o fato de o SQR não testar uma variante **não prova** que ela inexiste na literatura"):

1. **(G1)** Existe na literatura análise de **viabilidade por link budget** para cenários de RAN slicing simulado? Outros trabalhos reportam calibração de potência/raio/antena antes de avaliar controladores? Que cenários calibrados são considerados "realistas" (potência, raio, antena, LOS)?
2. **(G2)** Que **protocolos de reprodutibilidade** são praticados/propostos em slicing+RL (registro de premissas, código aberto, seeds/IC)? Algum trabalho anterior fez auditoria de reprodutibilidade de artigo de slicing com RL? (Se houver precedente, nosso diferencial precisa ser refinado.)
3. **(G3)** Alguém **instrumenta a aplicação da ação no MAC** (cota de PRBs realmente agendada, RBGs ociosos) em vez de assumir que a ação chega ao scheduler? Como a literatura valida o canal ação→efeito?
4. **(G4)** Quais **métricas de economia de recursos** são usadas (denominadores, agregações) e com que suporte estatístico (seeds, ICs, testes)?
5. **(G5)** Que trabalhos validam **xApps reais sobre E2/RIC** (E2AP/E2SM) com simulação, com que paridade/latência reportadas? Existe comparação IPC vs E2 na literatura? O que já foi feito com NORI/ns-O-RAN/FlexRIC (evitar reinvenção e citar a linhagem)?
6. **(G6)** A literatura caracteriza **regimes de carga/viabilidade** (factível × marginal × infactível) para controladores de slice? Existe métrica de diagnóstico equivalente à nossa diversidade de modos? Quem estuda degradação gracefully sob sobrecarga?
7. **(G7)** Como os trabalhos harmonizam **SLA per-UE vs per-slice** ao comparar métodos com granularidades distintas? Qual métrica primária comum é reportada?
8. **(G8)** Existe **coordenação conjunta (inter-slice) consciente de risco** já publicada (o candidato a novidade do SQR-JC)? Qual o SOTA em safe/constrained/distributional RL para alocação de PRBs, e que baselines são considerados justos (KBRL, DRL modernos, RSLAQ)? A projeção proporcional e suas falhas (ação segura local → insegura global) já foi analisada?

**Critério de uso dos resultados:** cada resposta vira uma linha da `literature-matrix.csv` com DOI verificado; a novidade do artigo novo é declarada **apenas** contra o que a matriz demonstrar que existe (regra do `GUIA` §12: poucos resultados de busca não provam lacuna).

## 8. Avisos metodológicos para as fases seguintes

- **Lacuna de busca ≠ lacuna científica** (regra da skill e do `GUIA` §12): antes de declarar G1–G8 como não atendidos pela literatura, expandir sinônimos, snowballing (para trás e para frente) e inspecionar related work dos incluídos.
- **Não inferir full-text de abstract**: papers só entram na matriz com texto completo avaliado; abstract-only fica no `screening-log.csv`.
- **Números deste documento**: todos os nossos são de artefatos locais versionados (`README.md`, `assumptions_register.md`, `fase5.4-xapp-sqr-paridade.md`, `summary_by_method.csv` — este último lido integralmente nesta sessão); os claims A1–A5 do artigo são **transcrições do artigo**, não fatos independentemente verificados.
- O artigo novo **não é escrito aqui**: a skill exige aprovação do pesquisador após a matriz de evidências antes de qualquer redação.
