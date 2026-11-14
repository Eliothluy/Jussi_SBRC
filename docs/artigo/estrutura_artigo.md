# Estrutura do artigo novo — título, abstract, venues e mapa seção a seção

> **Skill:** oran-literature-review · **Artefato:** pós-posicionamento (`estrutura_artigo.md`) · **Data:** 2026-10-18
> **Entradas lidas nesta sessão:** `docs/artigo/posicionamento.md` (integral — C1–C3 + N1), `docs/literature/literature-matrix.csv` (parse integral: 60 linhas L02–L61, DOIs conferidos campo a campo), `docs/literature/evidence-report.md` (§1–§8), `docs/literature/research-gaps.md` (G1–G8 + N1–N3), `docs/literature/work-understanding.md` (§1–§8), `README.md` (§1–§6), `MODA/docs/sqr/assumptions_register.md` (v1.3, R01–R31), `MODA/docs/sqr/fase5.4-xapp-sqr-paridade.md` (§1–§6), `MODA/ns-o-ran-gym/results/sqr_campaign/tables/{summary_by_method,modes_by_method,violation_by_slice}.csv` (lidos integralmente nesta sessão), listagem de `MODA/ns-o-ran-gym/results/sqr_campaign/figures/` (5 PNGs) e de `MODA/ns-3-dev/scratch/sqr/`.
> **Convenção de citação:** `[Lxx]` = linha de `literature-matrix.csv` + DOI (regra anti-alucinação da skill: só entram aqui os 60 artigos verificados na matriz; campo ausente = "não reportado"). Evidência própria = `[EQUIPE]`, com caminho:linha do artefato.
> **Status:** documento de planejamento de estrutura — **não é manuscrito**. Conforme a skill, nenhuma redação do artigo começa antes da revisão e aprovação explícita do pesquisador.
> **Datas de venues verificadas hoje (2026-10-18):** SBRC 2027 via `sbrc.sbc.org.br/2027/index.php/trilha-principal/` (WordPress REST, página 874); ICNS3 via `nsnam.org/research/icns3/icns3-2026/call-for-papers` e `easychair.org/cfp/icns3-2025`; SBrT via `sbrt2026.sbrt.org.br`. CFPs de 2027 do ICNS3/SBrT ainda não publicados (verificado hoje) — prazos marcados como **esperados** com base nas edições anteriores.

---

## 1. Título provisório (3 alternativas)

| # | Título | Ênfase | Prós | Contras |
|---|---|---|---|---|
| **A** | *Safe RAN slicing sob incerteza de viabilidade: auditoria empírica, cota verificável no MAC e validação E2 de controladores quantílicos* | As 3 contribuições C1+C2+C3 | Derivado do título do README §4 (`README.md:80`) — **não herdado literal**: os dois últimos termos da cadeia foram trocados ("calibração realista"→"cota verificável no MAC"; "validação O-RAN"→"validação E2") para nomear os diferenciais C2/C3 *[CORRIGIDO 2026-10-18: a proveniência anterior dizia "Herdado", mas o título do README lê "…auditoria empírica, calibração realista e validação O-RAN de controladores quantílicos"]*; cobre o escopo completo; "cota verificável" e "validação E2" são os diferenciais C2/C3 | Longo; "auditoria" pode assustar quem espera artigo de método |
| **B** | *Quando o cenário publicado é inviável: auditoria de controladores safe RAN slicing por sweep físico, regimes e paridade IPC×E2* | A descoberta central (C1 + G6) + C3 | Pergunta direta; destaque para o achado de regime (inviável ⇒ γ=0 ≈ γ=1); original | Menos visível a contribuição de ferramental (C2); "paridade IPC×E2" é jargão para leitor geral |
| **C** | *Da ação ao agendamento, do IPC ao E2: verificação fim a fim de controladores de slicing O-RAN em simulador calibrado* | Os dois protocolos de verificação (C2+C3) | Contribuições metodológicas em evidência; forte para a comunidade ns-3/O-RAN (venue 2) | Subordina a auditoria C1 — que é o resultado mais forte — a "verificação" |

**Recomendação:** **A** como título base (alinhado ao README §4 e ao posicionamento C1>C2>C3); C é a alternativa para a venue ICNS3 se o recorte migrar para o ferramental.

---

## 2. Abstract esqueleto (~150 palavras, placeholders em ⟨⟩)

> Controladores de RAN slicing com aprendizado por reforço são avaliados em cenários cuja viabilidade física não é verificada; o SQR reporta violações de SLA <1% para o seu. Reproduzimos fielmente o SQR — regressão quantílica online com kernel Matérn, sem rede neural — em ns-3/5G-LENA e auditamos seu cenário publicado: com os parâmetros publicados (raio 2 km, 30 dBm), ⟨X⟩% de violação para os quatro métodos avaliados (⟨N⟩ seeds pareadas, IC 95%), com ⟨Y⟩% das janelas sob entrega nula por RLF de borda. Um sweep controlado de raio (0% a 300 m → ⟨Z⟩% a 2 km) separa falha de implementação de inviabilidade de parâmetro, quantificada pela análise de link budget (déficit de ⟨D⟩ dB). Propomos um protocolo de verificação ação→agendamento (cota absoluta de RBGs + telemetria de ociosidade) e a primeira paridade de caminho de controle IPC×E2 bit a bit (⟨P⟩/⟨P⟩ decisões idênticas; p95 ⟨L⟩ ms) para um controlador de slicing sobre E2SM-KPM/RIC Control. Os resultados caracterizam regimes viável, marginal e inviável — os efeitos claimados só existem no primeiro. Registro de premissas, código e dados abertos.

Valores já conhecidos para os placeholders (fonte: `summary_by_method.csv:52,103,154,205`; `README.md:38,44-48`; `fase5.4-xapp-sqr-paridade.md:54,68`): X≈84–90; N=10; Y=56,6; Z=88; D≈25–30; P=3585; L=9,28. Multi-seed do sweep (RQ1.2) pode alterar Z — rodar antes de fechar o abstract.

*[CORRIGIDO 2026-10-18, sessão de revisão: a 1ª frase anterior — "Controladores de RAN slicing com aprendizado por reforço reportam violações de SLA <1%…" — atribuía o número à classe; na matriz, "<1%" é claim apenas de [L55] (grep no `literature-matrix.csv`: única ocorrência em `key_results` de L55; [L23] alega "almost negligible", sem número). O número ficou reatribuído ao artigo auditado.]*

---

## 3. Público-alvo e venues candidatas

### 3.1 Público-alvo

1. **Primário — comunidade de redes/O-RAN (SBC e internacional):** pesquisadores e engenheiros de gerenciamento de redes celulares 5G/6G, slicing, RIC/xApps e IA aplicada a redes; avaliadores de artigos de RL para RAN que precisam de critérios de validade (calibração, cota, estatística).
2. **Secundário — comunidade ns-3/5G-LENA:** desenvolvedores/usuários do simulador interessados em calibração, instrumentação de scheduler e integração E2 (linhagem WNS3/ICNS3 — 7 entradas da matriz: L38–L42, L44, L46, L49).
3. **Terciário — comunidade de reprodutibilidade em networking:** interessada em auditorias/replicações com protocolo de rastreabilidade `[ARTIGO]/[ASSUMIDO]` (registro de 31 linhas, `assumptions_register.md`).

### 3.2 Venue candidata 1 — SBRC 2027 (Trilha Principal) ⭐ recomendação de primeira submissão

| Item | Detalhe (verificado hoje, 2026-10-18) |
|---|---|
| Evento | 45º Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos — **17 a 21 de maio de 2027, Foz do Iguaçu-PR, presencial** (UTFPR/UNIOESTE/UFPR) |
| Prazos | **Registro de artigos: 11/01/2027 · Submissão: 18/01/2027 · Resultados: 15/03/2027 · Versão final: 26/03/2027** (fonte: `sbrc.sbc.org.br/2027/index.php/trilha-principal/`) |
| Histórico | ~300 submissões/ano; **30–35% de aceitação** (mesma página) |
| Tópicos que encaixam (lista da CFP) | "Redes celulares (5G, 6G) e suas aplicações" · "Inteligência Artificial/Aprendizado de Máquina em redes de computadores e sistemas distribuídos" · "Gerenciamento, operação, projeto e análise de redes" · "Plataformas experimentais de redes" · "Qualidade de serviço (QoS) e de experiência (QoE)" · "Desempenho, escalabilidade e confiabilidade" |
| Justificativa | (i) Escopo exato: C1 (análise de redes/gerenciamento) + C2 (plataformas experimentais) + C3 (5G/QoS); (ii) ecossistema nacional O-RAN relevante — o NORI [L12]+10.14209/sbrt.2025.1571157244 é do LASSE/UFPA e nosso port o estende; (iii) prazos forçam concluir os pré-requisitos (§8) já em janeiro; (iv) notificação 15/03/2027 deixa janela limpa para re-submissão à venue 2 se rejeitado |
| Risco | Formato SBC (páginas limitadas) pode comprimir as 3 contribuições; mitigação: C1 como eixo narrativo, C2/C3 como protocolos (tabelas compactas) |

### 3.3 Venue candidata 2 — ICNS3 2027 (International Conference on ns-3, ex-WNS3)

| Item | Detalhe (verificado hoje, 2026-10-18) |
|---|---|
| Evento | ICNS3 — sucessora direta do Workshop on ns-3 (WNS3); ICNS3 2026 ocorreu **19–21/10/2026, Nápoles** (`nsnam.org/research/icns3/icns3-2026/call-for-papers`); **CFP 2027 ainda não publicado** |
| Prazo | **Esperado ~março–abril/2027** com base no padrão: ICNS3 2025 — deadline original 21/03/2025 (17:00 EST), estendida a 04/04/2025, evento 19–21/08/2025, Osaka (`easychair.org/cfp/icns3-2025`; `nsnam.org`). A confirmar quando o CFP sair |
| Proceedings | ACM (DOI 10.1145/3747204 — exatamente as linhas L41+10.1145/3747204.3747214 e L42+10.1145/3747204.3747211 da matriz) |
| Justificativa | (i) Nosso ambiente é ns-3.46-dev + 5G-LENA v5.2 — a comunidade que mantém e calibra essa stack; (ii) C1 usa a calibração 5G-LENA [L47]+10.1016/j.simpat.2022.102580 e o canal site-specific [L43]+10.1109/ccnc65079.2026.11366300 como instrumento — conversa direta com o público do venue; (iii) C2 (verificação de scheduler `SqrMacScheduler`) é ferramental reutilizável; (iv) o port NORI→ns-3.46/5G-LENA v5.2 (`contrib/nori`, 55 arquivos) interessa a quem integra E2 em ns-3 (linhagem L06+10.1145/3592149.3592161 — publicado no WNS3 '23); (v) 7 entradas da matriz são dessa linhagem — público natural e revisores familiarizados |
| Risco | Menos peso para a audiência de RL (C1 tem diagnóstico de controlador); mitigação: título C e ênfase em calibração/instrumentação |

### 3.4 Planos B (registro, não candidatas principais)

- **SBrT 2027** — SBrT 2026: submissões 25/03/2026 → prazo final 27/05/2026, evento 29/09–02/10/2026 (`sbrt2026.sbrt.org.br`); **SBrT 2027 não anunciado**; janela esperada ~março–maio/2027. Justificativa: mesmo venue do NORI [L12] (SBrT 2025) — canal natural da extensão NORI→C3 em pt/en curto. Risco: escopo mais telecom/processamento de sinais.
- **Revistas (versão estendida)**: *Simulation Modelling Practice and Theory* (Elsevier — onde publicaram L45+10.1016/j.simpat.2019.101933 e L47+10.1016/j.simpat.2022.102580, as âncoras de calibração do C1) ou IEEE Access (onde publicaram L51+10.1109/access.2020.3039423). Sem prazo — usadas como extensão pós-conferência com os dados multi-seed completos (RQ1.2/RQ5.2).

**Estratégia de submissão proposta:** SBRC 2027 (18/01/2027) → se rejeitado, ICNS3 2027 (~mar–abr/2027) → SBrT 2027 (~abr–mai/2027) → revista estendida. Cada transição reutiliza o núcleo C1–C3.

---

## 4. Estrutura seção a seção (argumento → fonte própria → slots de citação)

Formato: **§. O que argumenta · Fonte própria `[EQUIPE]` · Slots `[Lxx]+DOI` — o que apia**.

### §1 Introdução (≈1 coluna)
- **Argumento:** controladores de slicing com RL são avaliados em cenários cuja viabilidade física nunca é checada e cujas cadeias ação→execução→protocolo são assumidas; reproduzimos fielmente o SQR e a cadeia inteira colapsa (84–90% de violação para todos os métodos no cenário publicado) — o problema é do cenário, não do código; este artigo audita, torna a cota observável e valida o caminho de controle. Contribuições C1–C3 + N1 anunciadas ao final.
- **Fonte própria:** `README.md:36-49` (§3.1–3.2 resultados), `README.md:62-78` (§4, G1–G8), `posicionamento.md:25-30` (eleição C1–C3), `work-understanding.md:85-93` (claims A1–A5 do artigo auditado).
- **Slots:** [L55]+10.1109/ccnc65079.2026.11366589 — o objeto auditado (violação <1% alegada; O-RAN apenas conceitual); [L23]+10.1109/twc.2022.3195570 — baseline KBRL e único exemplar da amostra com seeds+IC; [L06]+10.1145/3592149.3592161 e [L12]+10.14209/sbrt.2025.1571157244 — linhagem ns-3↔E2 que nos precede (honestidade desde a intro); [L02]+10.2307/1913643 — base teórica da perda pinball (opcional, 1 frase).

### §2 Trabalhos relacionados (≈1,5 colunas) — 3 subseções
- **§2.1 Controladores de slicing com RL e suporte estatístico.**
  - **Argumento:** famílias de algoritmo coexistem com protocolos de validação frágeis (1/60 com seeds+IC; seleção pós-hoc "melhores 64 runs"); a combinação kernel+quantílico sem rede neural tem 1 antecedente (KBRL) e 1 instanciação (SQR); safe/constrained RL é campo populoso — citar sem reivindicar primazia.
  - **Fonte:** `evidence-report.md:66-79` (§2.3 famílias), `evidence-report.md:145-155` (§5 estatística), `research-gaps.md:150-153` (G8 antecedentes), `posicionamento.md:140` (N1).
  - **Slots:** [L23]+10.1109/twc.2022.3195570 (KBRL: 30 runs, IC 90/95%); [L55]+10.1109/ccnc65079.2026.11366589 (SQR: "múltiplas execuções" sem seeds/IC); [L24]+10.1109/jsac.2023.3336191 (seleção pós-hoc); [L25]+10.1109/icmlcn64995.2025.11140546 (SafeSlice — projeção p/ ação segura, análogo da Π); [L26]+10.1109/jsac.2023.3336156 (CMARS — coordenação multi-agente restrita); [L27]+10.1109/jsac.2020.3041405 (RAWS — inviabilidade como mecanismo de treino); [L59]+10.1109/icnp52444.2021.9651934 (CaDRL); [L56]+10.1145/3583780.3615043 e [L57]+10.1109/icnp49622.2020.9259378, [L58]+10.1109/bigdata52589.2021.9671840 (campo safe/constrained); [L30]+10.1109/jsac.2019.2959185, [L60]+10.1109/twc.2021.3107520, [L61]+10.1109/eucnc/6gsummit63408.2025.11037178 (quantílico/distributional neural); [L18]+10.1109/tmc.2026.3666787 (RSLAQ — baseline DRL citado e não avaliado pelo SQR); [L37]+10.1109/mecom67453.2025.11439664 (DORA — regime de congestionamento em stack real); [L28]+10.1109/ojcoms.2026.3699475, [L29]+10.1109/tmlcn.2025.3534139 (risk-aware — vizinhança temporal; full-text pendente).
- **§2.2 Calibração e fidelidade de simulação ns-3/5G-LENA.**
  - **Argumento:** o ferramental de calibração existe e é maduro — mas nunca é usado por trabalhos de controlador RL (gap transversal N3); a nossa auditoria usa a stack calibrada como instrumento de veredito sobre parâmetros de terceiros.
  - **Fonte:** `research-gaps.md:25` (família de calibração), `evidence-report.md:120` (classe 1), `evidence-report.md:171` (calibração via 3GPP como âncora).
  - **Slots:** [L47]+10.1016/j.simpat.2022.102580 (calibração 5G-LENA 3GPP IMT-2020 — método-âncora do C1); [L45]+10.1016/j.simpat.2019.101933 (ancestral NR calibrado — genealogia do contrib/nr); [L50]+10.5281/zenodo.6553700 (canal industrial InF); [L44]+10.1145/3389400.3389401 (canal espacial); [L43]+10.1109/ccnc65079.2026.11366300 (ray-tracing site-specific — caminho futuro); [L51]+10.1109/access.2020.3039423 (link budget com padrões efetivos — fundamento do item antena do §5.3); [L54]+10.1186/s13638-022-02133-3 (células grandes inviáveis — corroboração externa); ferramental scheduler 5G-LENA: [L38]+10.1145/3837358.3837362, [L39]+10.1145/3659111.3659118, [L40]+10.1145/3592149.3592159, [L41]+10.1145/3747204.3747214, [L42]+10.1145/3747204.3747211; tráfego: [L46]+10.1145/3532577.3532602 (opcional).
- **§2.3 Validação E2/xApps (o que nos precede — citar obrigatoriamente).**
  - **Argumento:** a validação E2 real está madura (8 entradas da matriz medem latência/QoS/CPU via E2); o que nenhuma reporta é **paridade** entre caminho local e caminho E2.
  - **Fonte:** `evidence-report.md:127-139` (§4.2, tabela coluna "Paridade IPC×E2? Não" nas 8 linhas), `research-gaps.md:99-101` (G5).
  - **Slots:** [L06]+10.1145/3592149.3592161 (ns-O-RAN); [L12]+10.14209/sbrt.2025.1571157244 (NORI — completamos o "trabalho futuro" declarado: canal de controle RIC→cota); [L11]+10.1145/3485983.3494870 (FlexRIC — rota avaliada e não adotada); [L14]+10.1109/icccn69946.2026.11662852 (latência dentro do RIC, sem loop fechado); [L15]+10.1109/iccworkshops67674.2025.11162144 (REAL — xApp PPO); [L31]+10.1109/icnc59896.2024.10556357 (DRL Colosseum); [L20]+10.1109/tnsm.2026.3733344 (xDevSM — 4,5–7 ms, cota rule-based); [L16]+10.1145/3636534.3701544 (ORANSlice — cota via E2 em hardware); [L07]+10.1109/tmc.2022.3188013 (ColO-RAN — ação em PRBs sem verificação de execução no MAC); plataforma/survey: [L04]+10.1109/dyspan53946.2021.9677430 (Colosseum), [L21]+10.1109/comst.2023.3239220 (survey O-RAN), [L13]+10.1109/mcomstd.0001.2200041 (latência de interfaces O-RAN — orçamento, opcional); integrações: [L08]+10.1109/tmc.2023.3266642, [L09]+10.48550/arxiv.2208.14877, [L10]+10.1109/ictc66702.2025.11388719, [L19]+10.1109/milcom58377.2023.10356316, [L22]+10.1109/icoin59985.2024.10572115 (conforme cabimento; não todas obrigatórias).

### §3 O método auditado e o problema (≈1 coluna)
- **Argumento:** formulação publicada do SQR (estado z/ação a∈{0..C}/custo γ·a/C/projeção proporcional/adaptação τ-M), o par SQR×KBRL e por que a comparação original não é reproduzível (simuladores distintos, granularidades de SLA distintas, simulador do SQR não publicado).
- **Fonte:** `work-understanding.md:17-29` (§2–§3), `evidence-report.md:104-112` (§3.2 tabela SQR×KBRL), `assumptions_register.md` (R01:19, R08:26, R11:29, R18:36, R19:37, R24:42, R25:43), `MODA/docs/sqr/source_traceability.md` (âncoras [ARTIGO] seção/tabela).
- **Slots:** [L55]+10.1109/ccnc65079.2026.11366589 (formulação, Tabela I); [L23]+10.1109/twc.2022.3195570 (KBRL per-slice, garantia algébrica Σa≤C); [L02]+10.2307/1913643 (pinball); [L25]+10.1109/icmlcn64995.2025.11140546 (projeção análoga — contexto G8, citar ao discutir a Π).

### §4 Metodologia de auditoria (≈1,5 colunas) — 4 subseções
- **§4.1 Ambiente MODA.** ns-3.46-dev + 5G-LENA v5.2; `SqrMacScheduler` (cota absoluta de RBGs); loop Python↔ns-3 de 100 ms; cenário fiel à Tabela I (C efetivo = 53 PRBs lido da telemetria, R01).
  - **Fonte:** `README.md:11-22` (§1), `assumptions_register.md:19` (R01), `MODA/docs/sqr/contrato_interface.md`, `scratch/sqr/sqr-mac-scheduler.{h,cc}`.
  - **Slots:** [L06]+10.1145/3592149.3592161 (ancestral ns-O-RAN Gym); [L49]+10.1145/3345768.3355942 (SEM — execução gerenciada, contexto de ferramenta, opcional); [L22]+10.1109/icoin59985.2024.10572115 (O-RAN simulator com 5G-LENA — antecedente de integração, opcional).
- **§4.2 Rastreabilidade `[ARTIGO]/[ASSUMIDO]` e port fiel do KBRL.** 31 linhas R01–R31; 12+ parâmetros não publicados assumidos com justificativa; KBRL portado do repositório oficial (commit 65a3891).
  - **Fonte:** `assumptions_register.md` (integral), `MODA/docs/sqr/source_traceability.md`, `research-gaps.md:44-47` (G2).
  - **Slots:** [L55]+10.1109/ccnc65079.2026.11366589 (as omissões catalogadas); [L23]+10.1109/twc.2022.3195570 (código público do baseline — rodapé Sec. VII); [L07]+10.1109/tmc.2022.3188013 (promessa de artefato não cumprida — contraste); plataformas abertas: [L12], [L16], [L20], [L49] (o ecossistema tem infra para reprodutibilidade).
- **§4.3 Protocolo estatístico.** Campanha pareada 4 métodos × 10 seeds × 600 s; IC 95% por bootstrap na unidade seed; métrica primária per-UE pré-declarada + secundária per-slice (denominadores explícitos UE-tempo × slice-tempo).
  - **Fonte:** `assumptions_register.md:41` (R23), `:37` (R19), `:44` (R26); `results/sqr_campaign/tables/per_seed_metrics.csv`; `research-gaps.md:134-136` (G7).
  - **Slots:** [L23]+10.1109/twc.2022.3195570 (padrão-ouro igualado); [L55]+10.1109/ccnc65079.2026.11366589 e [L24]+10.1109/jsac.2023.3336191 (o que superamos); [L17]+10.1109/tmc.2024.3476338, [L05]+10.1109/tcomm.2024.3519550 (granularidades inter/intra-slice — motivação do duplo nível, G7).
- **§4.4 Gate de verificação ação→agendamento (R27).** Semântica absoluta de cota (RBGs excedentes ociosos, sem renormalização); gates adversariais Σa=0 / 0,5·C / C; telemetria `prb_requested/executed/scheduled/idle`.
  - **Fonte:** `assumptions_register.md:45` (R27 — critérios i–iv validados, 300 janelas), `scratch/sqr/sqr-mac-scheduler.cc` (`ComputeRbgQuotas`), `scratch/sqr/smoke_test.sh`.
  - **Slots:** [L07]+10.1109/tmc.2022.3188013 ("sem verificação de execução no MAC" — motivação direta); [L18]+10.1109/tmc.2026.3666787 (simulador sem MAC detalhado); [L23]+10.1109/twc.2022.3195570 (garantia algébrica não sobrevive ao mapeamento %→RBG com C=53); [L37]+10.1109/mecom67453.2025.11439664 e [L16]+10.1145/3636534.3701544 (execução real por construção — não publicam protocolo adversarial; claim ajustado, `posicionamento.md:99`).

### §5 Resultados I — Auditoria do cenário publicado (C1; ≈2 colunas) — 4 subseções
- **§5.1 Campanha no cenário publicado (Figs. 2–4).** Violação per-UE 84,27 [80,37–88,04] (equal), 88,89 [86,82–90,90] (KBRL), 89,64 [86,53–92,20] (SQR γ=0), 89,94 [86,74–92,50] (SQR γ=1); per-slice 84,93–89,76%; consumo 53,0 (equal) vs 27,4–28,0 PRBs (SQR); 56,6% das janelas com entrega nula; ~2,7 quedas RLF/UE/600 s.
  - **Fonte:** `summary_by_method.csv:52,103,154,205` (per-UE), `:43,94,145,196` (per-slice), `:16-19,118-121,169-172` (PRBs), `violation_by_slice.csv`, `figures/fig{2,3,4}_*.png`, `README.md:38`, R28 (`assumptions_register.md:47`).
  - **Slots:** [L55]+10.1109/ccnc65079.2026.11366589 (claims <1% e ">60%" vs "20%" — contraste qualitativo, NUNCA numérico cross-paper, regra `evidence-report.md:161-175`); [L18]+10.1109/tmc.2026.3666787 (outage com recursos insuficientes — fenômeno análogo, sem RLF modelado).
- **§5.2 Sweep de raio — o discriminador implementação×parâmetro.** Mesmo código, só o raio muda: 0,0% @300 m → 9,7% @800 m → 58,7% @1500 m → 88,0% @2 km.
  - **Fonte:** tabela em `README.md:42-49` (§3.2) e `work-understanding.md:50-57`. **Figura a produzir** para o manuscrito: os runs do sweep não estão consolidados em `results/` (busca por `*sweep*` em `results/` e `smoke_out/` sem resultado nesta sessão — registrar/organizar os runs é pré-requisito, §8).
  - **Slots:** [L47]+10.1016/j.simpat.2022.102580, [L45]+10.1016/j.simpat.2019.101933 (calibração como âncora); [L54]+10.1186/s13638-022-02133-3 (corroboração externa da fronteira ~0,8–1,2 km); [L43]+10.1109/ccnc65079.2026.11366300 (ray-tracing como caminho futuro de calibração fina); [L37]+10.1109/mecom67453.2025.11439664 (estudo de regime mais próximo — por carga, não por física).
- **§5.3 Link budget — por que o parâmetro publicado não fecha.** Déficit 25–30 dB: antena 1×1 omni "15 dBi" vs array 8×8 (~15–20 dB); UMa NLOS p/ todos (~9 dB vs TS 36.942); 30 vs 43 dBm (~13 dB); nota TDD (~2,3×).
  - **Fonte:** `README.md:51-60` (§3.3, tabela 4 itens), R15/R16 (`assumptions_register.md:33-34`), R30 (`:50`).
  - **Slots:** [L51]+10.1109/access.2020.3039423 (padrões nominais superestimam — fundamento); [L54]+10.1186/s13638-022-02133-3 (ISD grande inviável); [L55]+10.1109/ccnc65079.2026.11366589 (os parâmetros publicados auditados).
- **§5.4 Regimes e diagnóstico interno (viável × marginal × inviável).** 100% modo incerto (`modes_uncertain_group_pct` = 100,0 [100–100] em γ=0 e γ=1), `A_safe` vazio, τ no piso, M no teto ⇒ γ=0 ≈ γ=1; diversidade de modos como métrica de diagnóstico.
  - **Fonte:** `summary_by_method.csv:115,166`, `modes_by_method.csv`, `figures/fig_modes.png`, `README.md:74` (G6), `research-gaps.md:116-119` (G6).
  - **Slots:** [L37]+10.1109/mecom67453.2025.11439664 (congestionamento deliberado — sem taxonomia nem diagnóstico interno); [L18]+10.1109/tmc.2026.3666787 (cenários de carga discretos); [L27]+10.1109/jsac.2020.3041405 (inviabilidade como mecanismo de treino, não como regime caracterizado); [L28]+10.1109/ojcoms.2026.3699475 e [L61]+10.1109/eucnc/6gsummit63408.2025.11037178 (instrumentos quantílicos de cauda que não são usados para diagnóstico — opcional).

### §6 Resultados II — Cota verificável no MAC (C2; ≈1 coluna)
- **Argumento:** onde a cota não é verificada, "economia de RBs" não significa nada; gate R27 i–iv + telemetria tornam γ·a/C observável; na campanha, 74,96 [65,62–83,53] (γ=0) e 76,75 [66,48–86,32] (γ=1) PRBs ociosos/slot vs 0,0005 sob equal.
- **Fonte:** `assumptions_register.md:45` (R27 validado), `summary_by_method.csv:17,119,170` (idle), `scratch/sqr/sqr-mac-scheduler.cc`, `posicionamento.md:73-99` (tabela de novidade C2 — converter em tabela do artigo).
- **Slots:** [L55]+10.1109/ccnc65079.2026.11366589 (ação a_i com Σa≤C sem demonstrar execução; ">60%" sem denominador); [L23]+10.1109/twc.2022.3195570 (garantia algébrica); [L07]+10.1109/tmc.2022.3188013, [L18], [L31]+10.1109/icnc59896.2024.10556357 (cadeia assumida, não instrumentada); [L37]+10.1109/mecom67453.2025.11439664 (cota em OAI real por construção, sem gate público); [L16]+10.1145/3636534.3701544 (cota seguindo MaxRatio via E2 real — antecedente mais próximo, sem verificação adversarial); [L38]+10.1145/3837358.3837362, [L39]+10.1145/3659111.3659118 (scheduler 5G-LENA instrumentado sem controlador externo); [L20]+10.1109/tnsm.2026.3733344 (cota rule-based verificada em contadores — mecanismo alternativo).

### §7 Resultados III — xApp E2 e paridade IPC×E2 (C3; ≈1,25 colunas)
- **Argumento:** não somos o primeiro xApp E2 sobre ns-3 — reivindicamos a primeira **paridade de caminho de controle**: mesmo agente, mesmo run, decisões bit a bit idênticas por IPC e por E2 (e2sim E2AP v3.01 + E2SM-KPM + RIC Control via NORI), com delta de latência p50/p95. Escopo DEGRADED declarado no próprio claim.
- **Fonte:** `fase5.4-xapp-sqr-paridade.md:54` (paridade 3585/3585; 1199/1199; 1213; 994=994; 0 timeouts), `:64-73` (latência: E2 fim a fim p50 2,39 / p95 9,28 / máx 12,05 ms; +1,23 ms vs IPC; orçamento 50 ms), run `results/sqr_e2_parity/run_20261014_190646_sim120s_seed1_g1/`, `scratch/sqr/sqr-e2-adapter.{h,cc}`, `examples/xapp_sqr.py`.
- **Slots:** [L06]+10.1145/3592149.3592161 (ns-O-RAN — medição única sem repetições); [L12]+10.14209/sbrt.2025.1571157244 (NORI — E2AP v2.02.03, E2SM-RC deixado como trabalho futuro, que completamos com RIC Control→cota); [L11]+10.1145/3485983.3494870 (mede quão rápido; nós medimos se decide igual); [L14]+10.1109/icccn69946.2026.11662852 (0,5–4 ms dentro do RIC — consistência de ordem de grandeza do nosso p95); [L20]+10.1109/tnsm.2026.3733344 (4,5–7 ms multi-stack); [L15]+10.1109/iccworkshops67674.2025.11162144, [L31]+10.1109/icnc59896.2024.10556357 (xApps RL reais via E2); [L16]+10.1145/3636534.3701544 (cota via E2 em hardware); [L55]+10.1109/ccnc65079.2026.11366589 ("suitable for deployment in an O-RAN RIC", Seção I, sem demonstração *[CORRIGIDO 2026-10-18: a frase "compatível com near-RT RIC" citada antes não consta no PDF — pdftotext: 0 ocorrências de "near-RT"/"compat"; texto real no bullet de contribuições]* — transformado em evidência mensurada).

### §8 Discussão: limitações e ameaças à validade (≈1 coluna)
- **Argumento:** o que explica a divergência com o publicado (física, não código — sweep como discriminador); o que não é comparável cross-paper (tabela de comparabilidade); limitações próprias declaradas antes que o reviewer as aponte.
- **Fonte:** `evidence-report.md:161-175` (§6.1/§6.2 — regra "não comparável quantitativamente"), `posicionamento.md:154-161` (§8 avisos), `research-gaps.md:189-194` (avisos de busca).
- **Slots:** [L24]+10.1109/jsac.2023.3336191 (seleção pós-hoc — por que ICs importam); [L43]+10.1109/ccnc65079.2026.11366300 e [L44]+10.1145/3389400.3389401 (canais mais ricos como trabalho futuro); [L28]+10.1109/ojcoms.2026.3699475, [L48]+10.1145/3834582.3834591 (vizinhos 2026 possivelmente não indexados — checar full-texts antes da submissão).

### §9 Trabalho futuro (≈0,5 coluna)
- **Argumento:** extensões com evidência zero não entram como contribuição: SQR-JC (coordenação conjunta consciente de risco — RQ8.1–8.3), heterogeneidade eMBB+URLLC+mMTC, RIC real em k8s (OSC Release I), matriz de calibração realista (8×8/43 dBm/LOS) e sweep multi-seed (RQ1.2).
- **Fonte:** `posicionamento.md:146-151` (§7), `research-gaps.md:155-160` (RQ8.1–8.3), `research-gaps.md:32-33` (RQ1.1–1.2), `MODA/third_party/nori_infra/preflight.md` (passos RIC real, referenciado em AGENTS §8).
- **Slots:** [L25]+10.1109/icmlcn64995.2025.11140546 (projeção SafeSlice); [L26]+10.1109/jsac.2023.3336156 (CMARS); [L59]+10.1109/icnp52444.2021.9651934 (CaDRL) — antecedentes que limitam a novidade do SQR-JC; [L15]+10.1109/iccworkshops67674.2025.11162144, [L37]+10.1109/mecom67453.2025.11439664, [L18]+10.1109/tmc.2026.3666787 (multi-serviço em stack real/MATLAB); [L12]+10.14209/sbrt.2025.1571157244 (RIC OSC Release I — modo FULL).

### §10 Conclusão (≈0,5 coluna)
- **Argumento:** síntese C1–C3; a mensagem transferível: "calibre antes de avaliar, verifique antes de claimar, compare caminhos antes de confiar" — com o registro de premissas e os gates como entregáveis reutilizáveis.
- **Fonte:** `README.md:80` (síntese do artigo proposto), `posicionamento.md:136-142` (N1).
- **Slots:** nenhum novo; re-citar [L47], [L23], [L12] em 1 frase cada no máximo.

---

## 5. Mapa de figuras e tabelas

| # | Artefato | Status | Seção | Novidade vs matriz (60 entradas) |
|---|---|---|---|---|
| Fig. 1 | Diagrama do ambiente/loop de controle (IPC + E2, `SqrMacScheduler`, xApp) | **A desenhar** (base: `README.md:11-18`, `fase5.4-xapp-sqr-paridade.md:18-33`) | §4.1, §7 | — |
| Fig. 2 | Taxa de violação por janela (réplica da Fig. 2 do artigo) | ✅ `results/sqr_campaign/figures/fig2_window_violation_rate.png` | §5.1 | Comparação SQR×KBRL no **mesmo** ambiente físico (original usava simuladores distintos — `evidence-report.md:112`) |
| Fig. 3 | Violação acumulada (réplica da Fig. 3) | ✅ `fig3_cumulative_violation.png` | §5.1 | idem + dupla granularidade per-UE/per-slice (G7) |
| Fig. 4 | Distribuição de PRBs por decisão (réplica da Fig. 4) | ✅ `fig4_prb_distribution.png` | §5.1/§6 | Consumo com denominador explícito (G4) |
| Fig. 5 | **Sweep de raio** (violação × raio; 300 m→2 km) | **A produzir** — números em `README.md:42-49`; runs a consolidar (RQ1.2 multi-seed antes do manuscrito) | §5.2 | ⭐ **Nenhum trabalho da amostra faz sweep controlado de raio/potência como discriminador implementação×parâmetro de controlador RL** (G1, [BUSCA-LIM]; `research-gaps.md:24`) |
| Fig. 6 | Diversidade de modos / diagnóstico de regime (`fig_modes.png` + `modes_by_method.csv`) | ✅ (talvez converter em tabela compacta) | §5.4 | ⭐ **Métrica de diagnóstico do estado interno do agente (`A_safe` vazio 100%) sem antecedente na amostra** (G6, [BUSCA-LIM]; `research-gaps.md:117`) |
| Fig. 7 (opcional) | Barras-resumo com IC 95% (`fig_summary_violation_bars.png`) | ✅ | §5.1 ou §10 | 1/60 publica seeds+IC (N1; `evidence-report.md:149`) |
| Tab. 1 | Parâmetros do cenário auditado com tags [ARTIGO]/[ASSUMIDO] (condensado do registro R01–R31) | ✅ fonte `assumptions_register.md` | §3/§4.2 | ⭐ **Protocolo de rastreabilidade sem antecedente de auditoria de artigo de slicing+RL na amostra** (G2, [BUSCA-LIM]) |
| Tab. 2 | **Link budget** — 4 déficits (antena/NLOS/potência/TDD), total 25–30 dB | ✅ fonte `README.md:51-60` + R15/R16/R30 | §5.3 | ⭐ **Nenhum trabalho de controlador RL da amostra declara/valida link budget antes de avaliar** (G1/N3, [BUSCA-LIM]; `research-gaps.md:24`) |
| Tab. 3 | Resultados da campanha: violação per-UE e per-slice (média + IC 95%, n_seeds=10) × 4 métodos | ✅ fonte `summary_by_method.csv` (linhas citadas no §4/§5) | §5.1 | Dupla granularidade + IC por seed — contraste N1 |
| Tab. 4 | **Paridade IPC×E2**: decisões 3585/3585, telemetria 1199/1199, KPM 1213, violações 994=994, 0 timeouts | ✅ fonte `fase5.4-xapp-sqr-paridade.md:54` | §7 | ⭐ **Paridade de caminho de controle não reportada por nenhuma das 60 entradas** — tabela §4.2 do `evidence-report.md:127-139` mostra "Não" nas 8 linhas E2 vs nossa "Sim" (G5/N2, [BUSCA-LIM]) |
| Tab. 5 | Latência por caminho (IPC vs E2 xApp; p50/p95/máx; orçamento 50 ms) | ✅ fonte `fase5.4-xapp-sqr-paridade.md:64-73` | §7 | Delta IPC→E2 quantificado (antecedentes medem latência só de um caminho: [L14], [L20]) |
| Tab. 6 | Gate R27 i–iv (Σa=C ⇒ utilização 1,0000; Σa=0,5·C ⇒ 57% ocioso; Σa=0 ⇒ 0 RBG) | ✅ fonte `assumptions_register.md:45` | §6 | ⭐ **Protocolo de verificação ação→agendamento sem antecedente** (G3, [BUSCA-LIM]; [L37]/[L16] executam por construção, não publicam gate) |
| Tab. 7 (opcional) | Tabelas de novidade C1/C2/C3 vs trabalhos mais próximos | ✅ fonte `posicionamento.md:50-63,86-99,120-130` | §2 (condensar) | Mapa explícito de diferencial linha a linha |

**Resumo dos ⭐ (o que a matriz indica que nenhum trabalho anterior mostrou — todos com [BUSCA-LIM] e sujeitos à re-checagem de §8):** (1) auditoria de reprodutibilidade de artigo de slicing+RL com reimplementação fiel e campanha pareada (G2); (2) sweep físico controlado + link budget como pré-requisito de validade de cenário de controlador RL (G1/N3); (3) protocolo de verificação ação→agendamento com telemetria de ociosidade e gates adversariais (G3); (4) paridade de caminho de controle IPC×E2 bit a bit com delta de latência (G5/N2); (5) taxonomia de regime viável/marginal/inviável com diagnóstico do estado interno do agente (G6); (6) reporte duplo per-UE+per-slice com métrica primária pré-declarada (G7/N1).

---

## 6. Riscos de reviewer e mitigação, por seção

| Seção | Risco (fraqueza real do nosso estudo) | Mitigação no texto |
|---|---|---|
| §1 | "É só uma reprodução que falhou" — ausência de contribuição positiva | Enquadrar como auditoria com 3 protocolos transferíveis (C1–C3) + achado de regime; posicionar contra o déficit estatístico do campo (1/60 com seeds+IC) |
| §2 | Reviewer aponta trabalho safe/constrained ou de E2 que "já fez isso" | Citar preventivamente os antecedentes limitadores ([L25], [L26], [L59]; [L06], [L12], [L15], [L31], [L16], [L20]) com a tabela de novidade condensada; claims estreitos conforme `posicionamento.md` §3–§5 |
| §3 | Port não fiel ao método auditado | Registro `[ARTIGO]/[ASSUMIDO]` com âncoras de seção/tabela (`source_traceability.md`); R25 (γ alterna só o termo de custo, P_penalty ativa nas duas variantes); divergência com autores registrada como pergunta aberta (GUIA §15) |
| §4.2 | "Assumiram 12+ parâmetros — os resultados não dizem respeito ao método original" | Essa é a **tese** do G2, não uma fraqueza escondida: cada [ASSUMIDO] é justificado e entra na narrativa da irrepdutibilidade por omissão; sensibilidade prevista (R01 C∈{53,104}; R13 global vs por slice) |
| §5.1 | "Seu simulador está errado, não o artigo" | Sweep @300 m = 0% de violação (implementação autovalidada quando a física permite — `README.md:49`); calibração 5G-LENA contra 3GPP [L47]; RLF como causa raiz quantificada (56,6%; 2,7 quedas/UE) |
| §5.1 | "84–90% 'refuta' o <1%?" — confronto numérico cross-paper inválido | Regra do `evidence-report.md:166-175`: dizer explicitamente "não comparável quantitativamente"; o sweep é o discriminador implementação×parâmetro |
| §5.2 | Sweep com **1 seed por ponto** (network-only, 240 s) | Executar multi-seed (RQ1.2) **antes** da submissão e atualizar Fig. 5/abstract; se não houver tempo, declarar limitação com o plano de extensão — nunca apresentar 1 seed como fronteira quantificada (aviso `posicionamento.md:158`) |
| §5.3 | Link budget conservador demais / premissas de antena e ruído | Tags R15/R16/R30 (a leitura "Noise 9 dB" e "15 dBi" são do artigo; padrão de radiação e NF são nossos) — mostrar que mesmo a leitura mais favorável mantém déficit ≫10 dB |
| §5.4 | "γ=0 ≈ γ=1 significa que seu SQR está quebrado" | Apresentar como achado de regime: em inviabilidade o `A_safe` esvazia e o custo γ·a/C não discrimina; o sweep fornece o regime viável onde a separação deve ser testada (RQ6.1/6.2 como trabalho futuro) |
| §6 | "Gate só em simulação; [L37]/[L16] executam em stack real" | Claim já reescrito: protocolo de **verificação adversarial** + telemetria de ociosidade, não primazia de execução; apontar que execução por construção não é auditável/portável (`posicionamento.md:91`) |
| §7 | **Modo DEGRADED** — sem Near-RT RIC real; e2sim local | Declarar no claim e na limitação; p95 9,28 ms consistente com [L14] (0,5–4 ms) e [L20] (4,5–7 ms); RIC real (k8s, `MODA/third_party/nori_infra/preflight.md`) como pré-requisito ideal antes da submissão — se não ocorrer, o texto NÃO afirma "Near-RT RIC real" (aviso `posicionamento.md:159`) |
| §7 | "Paridade de 120 s × 1 seed é pouco" | Executar RQ5.2 (campanha via `--controlPath e2`, 10 seeds) ou estender paridade a mais seeds — script reutilizável (`run_sqr_e2_parity.sh`); senão declarar escopo (1 seed, γ=1, 120 s) |
| §8 | "Nenhuma comparação com DRL moderno" (G8) | Escopo: auditoria dos métodos do par original (SQR×KBRL); DRL moderno + SQR-JC como trabalho futuro com RQs definidos; a matriz mostra que nenhum trabalho da amostra mistura baseline kernel + DRL + estático no mesmo ambiente (`evidence-report.md:100`) |
| §8 | Re-checagem de novidade (33/60 entradas abstract-only; busca de 1 dia) | Reexecutar eixos D/B5 e buscar refutações ("projection failure", "action shadowing", "control path parity", "replication") em WNS3/ICNS3, ICMLCN, CoNEXT 2024–2026 antes da submissão (aviso `research-gaps.md:191`; `posicionamento.md:156`) |
| Global | Reprodutibilidade do **nosso** trabalho (o artigo critica omissão) | Publicar código + registro de premissas + scripts de campanha + tabelas (repo aberto no artigo); arquivos já versionados (`results/sqr_campaign/tables/`, `MODA/docs/sqr/`) |

---

## 7. Pré-requisitos antes da submissão (checklist herdado dos avisos)

1. **Sweep multi-seed** (RQ1.2: ≥10 seeds/ponto, mesmo código) → refazer Fig. 5 e abstract ⟨Z⟩.
2. **Consolidar runs do sweep em `results/`** (não localizados nesta sessão; busca `find -iname "*sweep*"` vazia em `results/` e `smoke_out/`).
3. **Paridade estendida** (RQ5.2: mais seeds/régua de carga) ou escopo declarado com precisão no §7.
4. **Decisão do usuário sobre RIC real** (k8s) — habilitaria modo FULL e removeria a maior reserva do C3.
5. **Re-busca de novidade** pré-submissão ([BUSCA-LIM] re-check; full-texts de [L28], [L48]).
6. **Release público** (código, registro, dados) + licença.
7. **Aprovação explícita do pesquisador** deste documento e do `posicionamento.md` (regra da skill — nenhum manuscrito antes disso).

---

*Gerado nesta sessão contra `literature-matrix.csv` (60 entradas verificadas, DOIs conferidos por parse integral), `posicionamento.md`, `evidence-report.md`, `research-gaps.md`, `README.md`, `assumptions_register.md` (v1.3), `fase5.4-xapp-sqr-paridade.md` e as tabelas de `results/sqr_campaign/`. Datas de venues verificadas em 2026-10-18 (SBRC 2027 exatas; ICNS3 2027 e SBrT 2027 esperadas — CFPs não publicados). Última atualização: 2026-10-18.*
