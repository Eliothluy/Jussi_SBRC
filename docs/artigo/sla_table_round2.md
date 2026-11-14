# Tabela de SLAs/KPIs citáveis — Rodada 2 (gate §8)

> **Protocolo**: `rodada_2_fronteira_viabilidade_sqr.md` §8 — "Não seguir para o sweep definitivo sem fechar essa tabela".
> **Data**: 2026-10-20 · **Agente**: Pesquisador-SLA (skill oran-literature-review)
> **Regra do jogo aplicada**: um alvo sem fonte aberta e verificada nesta sessão **não entra como alvo** — entra como "não reportado/candidato a assumir". Nenhum valor abaixo foi gerado de memória: cada linha tem o documento que foi aberto e lido nesta sessão (2026-10-20) ou o registro da tentativa frustrada.

## Legenda — nível de verificação

| Nível | Significado |
|---|---|
| **verificado no texto** | Full-text do documento lido nesta sessão (arXiv HTML, PDF ETSI/3GPP); número copiado da seção/tabela citada. |
| **claim do autor** | Afirmado pelo autor, sem acesso independente ao trecho (ex.: só abstract). |
| **não reportado / não verificado** | Não consta na parte acessível da fonte, ou a fonte não pôde ser aberta. |

---

## A. Tabela principal (alvos VERIFICADOS nesta sessão)

### A.1 — Alvos de artigos (nivel de experimento — âncoras citáveis da literatura de O-RAN/DRL)

| Slice | KPI crítico | Alvo numérico (com unidade) | Sentido | Prioridade | Fonte (artigo + seção/tabela) | Nível de verificação |
|---|---|---|---|---|---|---|
| URLLC | latência de pacote | **≤ 500 ms** (`t_target`, Eq. (2); Tabela I: "Latency (ms): 500") | menor é melhor | alta | Barker et al., REAL, IEEE ICC Workshops 2025, Tabela I + §III (arXiv:2502.00715v1) | **verificado no texto** (full-text arXiv HTML lido 2026-10-20) |
| URLLC | latência de pacote | **≤ 400 ms** (`L_tar`, Eq. (4) e §VI-B: "SLA latency target L_tar = 400 ms") | menor é melhor | alta | Ebrahimi Dorcheh et al., DORA, IEEE MECOM 2025, §III-A/Eq. (4) (arXiv:2509.07242v1) | **verificado no texto** (full-text arXiv HTML lido 2026-10-20) |
| eMBB | throughput por UE | **T_tar = 7 Mbps** (Eq. (5): "target eMBB SLA corresponding to the user-requested mean throughput"; demandas ~N(7, 0,7²) Mbps) | maior é melhor | média | DORA, §III-B/Eq. (5) (arXiv:2509.07242v1) | **verificado no texto** (full-text lido 2026-10-20) |
| eMBB | bitrate (alvo de recompensa) | **2×10⁵ – 4×10⁵** (Tabela I: "Bit Rate Min/Max"; unidade não explicitada na tabela — o registro de 18/10/2026 na literature-matrix interpreta como bps) | maior é melhor | média | REAL, Tabela I + Eq. (3) (arXiv:2502.00715v1) | **verificado no texto** (número verificado; unidade = claim do autor/registro) |
| mMTC | atendimento (critério de serviço) | **≥ 5 PRBs por dispositivo** (N_serv = ⌊PRB_mMTC/5⌋; §VI-B: "a device is serviced if assigned at least 5 PRBs") | maior é melhor | conforme caso | DORA, §IV-C/Eq. (6) (arXiv:2509.07242v1) | **verificado no texto** (critério definido pelos autores p/ o experimento; não é alvo normativo) |
| mMTC | modelo de entrega de payload | payload 25×10³–60×10³ B a 4 Hz (reward = bytes recebidos vs. esperados, Eq. (4)) — **modelo de tráfego, não SLA numérico** | maior é melhor | conforme caso | REAL, Tabela I + Eq. (4) (arXiv:2502.00715v1) | **verificado no texto** |
| URLLC | latência p95 / tail-risk | **não reportado** (o abstract define a restrição CVaR₀,₀₅ sobre a latência de cauda URLLC, mas não traz valor de SLA em ms) | menor é melhor | alta | Mahalakshmi & Markkandan, IEEE OJ-COMS 2026 (doi:10.1109/OJCOMS.2026.3699475) | **não reportado** (abstract completo lido via Semantic Scholar; full-text IEEE bloqueado — ver lacunas) |

**Observações sobre os alvos de artigo (importantes para honestidade do artigo novo):**
1. Os três artigos **não ancoram seus alvos em normas** — 400 ms (DORA) e 500 ms (REAL) são parâmetros de testbed, ordens de grandeza mais frouxos que os alvos URLLC normativos (0,5–1 ms, ver A.2). Nenhum dos dois cita 3GPP TS 22.261 nem ITU-R M.2410 como fonte do número.
2. DORA opera deliberadamente em regime de congestionamento (demanda agregada > 106 PRBs, §VI), o que explica a frouxidão do alvo URLLC.

### A.2 — Alvos normativos 3GPP (verificados abrindo o documento nesta sessão)

| Slice | KPI crítico | Alvo numérico (com unidade) | Sentido | Prioridade | Fonte (norma + seção/tabela) | Nível de verificação |
|---|---|---|---|---|---|---|
| URLLC | latência user plane | **0,5 ms UL / 0,5 ms DL** (§7.5: "For URLLC, the target for user plane latency should be 0.5 ms for UL, and 0.5 ms for DL") | menor é melhor | alta | 3GPP TR 38.913 v18.0.0 (38913-j00), §7.5 | **verificado no texto** (docx oficial 3GPP FTP, baixado e lido 2026-10-20) |
| URLLC | confiabilidade | **99,999 %** (§7.9: "A general URLLC reliability requirement for one transmission of a packet is 1−10⁻⁵ for 32 bytes with a user plane latency of 1 ms") | maior é melhor | alta | 3GPP TR 38.913 v18.0.0, §7.9 | **verificado no texto** |
| eMBB | latência user plane | **4 ms UL / 4 ms DL** (§7.5) | menor é melhor | média | 3GPP TR 38.913 v18.0.0, §7.5 | **verificado no texto** |
| eMBB | user experienced data rate | **Urbano macro: 50 Mbit/s DL / 25 Mbit/s UL; denso urbano: 300/50 Mbit/s; indoor hotspot: 1 Gbit/s DL / 500 Mbit/s UL** — Nota 5: "All the values in this table are targeted values and not strict requirements" | maior é melhor | média | 3GPP TS 22.261 v19.15.0 (ETSI TS 122 261 V19.15.0), Tabela 7.1-1, p. 88 | **verificado no texto** (PDF ETSI baixado e lido 2026-10-20) |
| URLLC (industrial) | disponibilidade + latência E2E | **99,999 % a 99,999 99 %** (por linha) e latência E2E **< intervalo de transferência** com intervalos de **500 µs / 1 ms / 2 ms** (controle de movimento, payloads 20–50 B) | maior/menor | alta | 3GPP TS 22.104 v19.2.0 (ETSI TS 122 104 V19.2.0), Tabela 5.2-1, p. 16 | **verificado no texto** (PDF ETSI baixado e lido 2026-10-20; é a norma que a TS 22.261 §7.2.2 aponta para KPIs URLLC) |
| URLLC (backhaul ITS) | latência E2E + disponibilidade + confiabilidade | **30 ms** máx. E2E; disponibilidade **99,9999 %**; confiabilidade **99,999 %**; user experienced rate 10 Mbit/s | menor/maior | alta (caso específico) | 3GPP TS 22.261 v19.15.0, Tabela 7.2.3.2-1, p. 91 | **verificado no texto** |
| mMTC | densidade de conexão | **1 000 000 dispositivos/km²** em ambiente urbano (§7.17: "The target for connection density should be 1 000 000 device/km² in urban environment") | maior é melhor | conforme caso (fora de escopo p/ 1 célula) | 3GPP TR 38.913 v18.0.0, §7.17 | **verificado no texto** (na TS 22.261 o mesmo valor aparece apenas como exemplo "e.g." nas cláusulas 6.4.2.1/6.4.2.2 — não como requisito) |

---

## B. Linhas NÃO fechadas (não usar como alvo citável)

| Slice | KPI crítico | Alvo | Sentido | Prioridade | Fonte | Nível de verificação |
|---|---|---|---|---|---|---|
| URLLC | latência user plane 1 ms / confiabilidade 99,999% / eMBB 4 ms / mMTC 10⁶ km⁻² "ITU" | — | — | — | ITU-R M.2410-0 ("Minimum requirements related to technical performance for IMT-2020") | **NÃO verificado — itu.int inacessível nesta sessão** (404 em 4 esquemas de URL testados; landing page "Publication selected is not available"; diretório dms_pubrec 403). Os valores citados de memória em muitos papers **não entram aqui** até o PDF ser aberto. Substituto verificado: TR 38.913 v18 (linhas A.2). |
| eMBB/URLLC/mMTC | SLA numérico por slice | — | — | — | CVaR-PPO, IEEE OJ-COMS 2026 (doi:10.1109/OJCOMS.2026.3699475) | **não reportado** — abstract completo (CC-BY, via Semantic Scholar API) não contém nenhum alvo numérico; full-text bloqueado por bot-detection IEEE (tentativas: doi.org→Xplore doc 11550086, PDF iel8, WebFetch direto = "blocked"). Consistente com literature-matrix.csv de 18/10 ("Xplore bloqueado"). |

---

## C. Proposta de adoção para a Rodada 2 (DECISÃO PENDENTE do orquestrador — marcada [ASSUMIDO] até aprovação)

A §8 do protocolo pede: URLLC latência (alta), URLLC PDR/reliability (alta), eMBB throughput (média), mMTC atendimento (conforme caso). Com o que está **verificado**:

| Linha §8 | Alvo proposto | Âncora citável | Status no assumptions_register |
|---|---|---|---|
| URLLC latência | **400 ms** (ou 500 ms) como SLA de referência do cenário de borda/congestionamento | DORA Eq. (4) / REAL Tabela I | `[ARTIGO]` se adotado como está; nota obrigatória de que é alvo de experimento, não normativo |
| URLLC PDR/reliability | proporção de pacotes entregues no deadline; alvo numérico tipo 99,999% (32 B, 1 ms) **inviável** no nosso cenário (2 km/30 dBm, RLF de borda, período de decisão 100 ms) | TR 38.913 §7.9 (norma, verificado) | qualquer número adotado aqui = `[ASSUMIDO]` com justificativa de inviabilidade do normativo |
| eMBB throughput | 7 Mbps por UE | DORA Eq. (5) | `[ARTIGO]` |
| mMTC atendimento | dispositivo atendido se receber ≥ 5 PRBs (critério direto do scheduler) | DORA §IV-C/Eq. (6) | `[ARTIGO]` (critério do autor) |

⚠️ A escolha 400 vs 500 ms muda o shape do sweep — não fechar sem go do orquestrador.

---

## D. Referências completas (identificadores verificados)

1. **REAL** — R. Barker, A. Ebrahimi Dorcheh, T. Seyfi, F. Afghah, "REAL: Reinforcement Learning-Enabled xApps for Experimental Closed-Loop Optimization in O-RAN with OSC RIC and srsRAN," *IEEE ICC Workshops*, 2025. DOI: [10.1109/ICCWorkshops67674.2025.11162144](https://doi.org/10.1109/ICCWorkshops67674.2025.11162144) · preprint: [arXiv:2502.00715v1](https://arxiv.org/html/2502.00715v1) (lido 2026-10-20). Tabela I "Task Specification for PRB Allocation"; Eqs. (2)–(4).
2. **DORA** — A. Ebrahimi Dorcheh, T. Seyfi, F. Afghah, "DORA: Dynamic O-RAN Resource Allocation for Multi-Slice 5G Networks," *2025 IEEE MECOM*. DOI: [10.1109/MECOM67453.2025.11439664](https://doi.org/10.1109/MECOM67453.2025.11439664) · preprint: [arXiv:2509.07242v1](https://arxiv.org/html/2509.07242v1) (lido 2026-10-20). §III-A/B Eqs. (4)–(5); §IV-C Eq. (6); §VI.
3. **CVaR-PPO** — S. Mahalakshmi, S. Markkandan, "CVaR-Constrained Optimization for Risk-Aware Dynamic Resource Allocation in 5G RAN Slicing," *IEEE OJ-COMS*, 2026. DOI: [10.1109/OJCOMS.2026.3699475](https://doi.org/10.1109/OJCOMS.2026.3699475) · abstract completo via Semantic Scholar (paperId 8840e1e2d8d3710ca25bfb979682fd93a2081d87, GOLD OA CC-BY) — sem alvos numéricos no abstract.
4. **3GPP TS 22.261 v19.15.0** — "Service requirements for the 5G system" (ETSI TS 122 261 V19.15.0, 2026-10, 119 p.). PDF: [etsi.org/deliver/.../19.15.00_60/ts_122261v191500p.pdf](https://www.etsi.org/deliver/etsi_ts/122200_122299/122261/19.15.00_60/ts_122261v191500p.pdf). Tabela 7.1-1 p. 88 (Nota 5: valores-alvo, não requisitos estritos); §7.2.2 p. 88–89 (URLLC delegado a 22.104/22.186/22.289); Tabela 7.2.3.2-1 p. 91; cláusulas 6.4.2.1–6.4.2.2 p. 26–27 ("e.g. 1 million/km²" = exemplo).
5. **3GPP TS 22.104 v19.2.0** — "Service requirements for cyber-physical control applications in vertical domains" (ETSI TS 122 104 V19.2.0, 2026-01, 49 p.). PDF: [etsi.org/deliver/.../19.02.00_60/ts_122104v190200p.pdf](https://www.etsi.org/deliver/etsi_ts/122100_122199/122104/19.02.00_60/ts_122104v190200p.pdf). Tabela 5.2-1 p. 16.
6. **3GPP TR 38.913 v18.0.0** — "Study on scenarios and requirements for next generation radio access technologies" (38913-j00.zip, 3GPP FTP, baixado 2026-10-20). §7.5, §7.9, §7.15, §7.17. [Fonte primária dos valores do ITU-R M.2410 no 3GPP; M.2410-0 em si NÃO aberto nesta sessão.]
7. **ITU-R M.2410-0** — "Minimum requirements related to technical performance for IMT-2020 radio interface(s)" (2017). **Não aberto** — todas as tentativas falharam (ver §E). Citação pendente de verificação.

---

## E. Registro de queries e tentativas (2026-10-20 — também em `docs/literature/tools/search-log.jsonl`)

| # | Ação / query | Resultado |
|---|---|---|
| 1 | webReader `arxiv.org/html/2502.00715v1` | full-text OK — Tabela I + Eqs. (2)–(4) |
| 2 | webReader `arxiv.org/html/2509.07242v1` | full-text OK — §III/IV/VI Eqs. (4)–(6) |
| 3 | WebFetch `doi.org/10.1109/OJCOMS.2026.3699475` → redirect ieeexplore doc 11550086 → re-fetch | **blocked** |
| 4 | WebSearch `"CVaR-Constrained Optimization" "Risk-Aware..." Mahalakshmi Markkandan` | achou S2/RG/PDF IEEE; levou à API S2 |
| 5 | curl IEEE `iel8/8782661/8901158/11550086.pdf` | HTML de challenge APM (bot-block), não PDF |
| 6 | curl Semantic Scholar API `DOI:10.1109/ojcoms.2026.3699475` | abstract completo (sem números de SLA) |
| 7 | curl ETSI dir `etsi_ts/122200_122299/122261/` → PDF v19.15.00 | OK (119 p.) → pdftotext → Tabelas 7.1-1, 7.2.3.2-1 |
| 8 | curl ETSI dir `122104/` → PDF v19.02.00 | OK (49 p.) → Tabela 5.2-1 |
| 9 | curl ITU `R-REC-M.2410-0-201709-I!!PDF-E.pdf` e variantes (201711, dms_pub/opb/rec, dms_pub/opb/spec) | 404 em todas |
| 10 | WebFetch `itu.int/pub/R-REC-M.2410` e `itu.int/rec/R-REC-M.2410/en`; curl dir `dms_pubrec/.../m/` | "not available" / página vazia / 403 |
| 11 | WebSearch `ITU-R M.2410-0 PDF download R-REC-M.2410` | só confirma URL padrão (que 404) |
| 12 | curl 3GPP FTP `Specs/archive/38_series/38.913/38913-j00.zip` → docx → extração de texto | OK — §7.5/§7.9/§7.17 verbatim |

## F. Lacunas (gaps)

1. **CVaR-PPO (OJ-COMS 2026)**: alvos numéricos de SLA por slice não verificáveis — full-text bloqueado (IEEE bot-detection; 3 tentativas registradas). Se necessário, pedir PDF aos autores ou checar via biblioteca. Até lá: "não reportado".
2. **ITU-R M.2410-0**: inacessível por automação nesta sessão — os valores canônicos "ITU" (1 ms URLLC / 4 ms eMBB / 99,999% / 10⁶ km⁻² / 100 Mbit/s DL) **não foram verificados** e não constam como alvo na Tabela B. Cobertura: TR 38.913 v18 (verificado) carrega os mesmos KPIs no 3GPP (§7.5/§7.9/§7.17), mas a citação "ITU-R M.2410, tabela X" segue proibida até abrir o PDF.
3. **REAL Tabela I**: unidade do "Bit Rate" eMBB (2×10⁵–4×10⁵) não explicitada no texto lido; bps é interpretação do registro de 18/10 (claim do autor/registro), não leitura direta.
4. **Nenhum dos 3 artigos ancora os alvos em normas** — para o artigo novo, isso é diferencial: citar TR 38.913/TS 22.261 como âncora normativa e justificar a divergência (cenário de borda/RLF) é contribuição de honestidade metodológica, não fraqueza.
5. **PDR/reliability para mMTC**: não há alvo numérico citável encontrado em nenhuma fonte aberta desta sessão — DORA usa critério de PRBs, REAL usa razão bytes recebidos/esperados. Qualquer % de PDR para mMTC será `[ASSUMIDO]`.
